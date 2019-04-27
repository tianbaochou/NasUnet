from torch.functional import F
from util.prim_ops_set import *
from util.genotype import *
from search.backbone.cell import Cell

class SearchULikeCNN(nn.Module):
    def __init__(self, input_c, c, num_classes, depth, meta_node_num=4,
                 double_down_channel=True, use_softmax_head = False):
        super(SearchULikeCNN, self).__init__()
        self._num_classes =  num_classes   # 2
        self._depth = depth
        self._meta_node_num = meta_node_num
        self._multiplier = meta_node_num
        self._use_softmax_head = use_softmax_head
        self._double_down_channel = double_down_channel

        c_prev_prev, c_prev, c_curr = meta_node_num * c, meta_node_num * c, c

        # the stem need a complicate mode
        # stem1 is a reduce block for stride 3x3 convolution
        # stem0 is a normal block for 1x1 convolution
        self.stem0 = ConvOps(input_c, c_prev_prev, kernel_size=1, ops_order='weight_norm')
        self.stem1 = ConvOps(input_c, c_prev, kernel_size=3,  stride=2, ops_order='weight_norm')

        assert depth >= 2 , 'depth must >= 2'

        self.down_cells = nn.ModuleList()
        self.up_cells = nn.ModuleList()

        down_cs_nfilters = []

        # create the encoder pathway and add to a list
        down_cs_nfilters += [c_prev]
        down_cs_nfilters += [c_prev_prev]
        for i in range(depth):
            c_curr = 2 *  c_curr if self._double_down_channel else c_curr  # double the number of filters
            down_cell = Cell(meta_node_num, c_prev_prev, c_prev, c_curr, cell_type='down')
            self.down_cells += [down_cell]
            c_prev_prev, c_prev = c_prev, self._multiplier*c_curr
            down_cs_nfilters += [c_prev]

        # create the decoder pathway and add to a list
        # Todo: the prev_prev channel and prev channel is the same for decoder pathway
        for i in range(depth+1):
            c_prev_prev = down_cs_nfilters[-(i+2)]
            up_cell = Cell(meta_node_num, c_prev_prev, c_prev, c_curr, cell_type='up')
            self.up_cells += [up_cell]
            c_prev = self._multiplier*c_curr
            # c_prev_prev, c_prev = down_cs_nfilters[-(i+2)] if self._double_down_channel else\
            #                           c_prev, self._multiplier*c_curr
            c_curr = c_curr // 2 if self._double_down_channel else c_curr  # halve the number of filters

        self.conv_segmentation = ConvOps(c_prev, num_classes, kernel_size=1, dropout_rate=0.1, ops_order='weight')

        if use_softmax_head:
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, weights1_down, weights1_up, weights2_down, weights2_up):
        s0, s1 = self.stem0(x), self.stem1(x)
        down_cs = []

        # encoder pathway
        down_cs.append(s0) # the s0 has original image size!!
        down_cs.append(s1)
        for i, cell in enumerate(self.down_cells):
            # Sharing a global N*M weights matrix
            # where M : normal + down
            s0, s1 = s1, cell(s0, s1, weights1_down, weights2_down)
            down_cs.append(s1)

        # decoder pathway
        for i, cell in enumerate(self.up_cells):
            # Sharing a global N*M weights matrix
            # where M : normal + up
            s0 = down_cs[-(i+2)] # horizon input
            s1 = cell(s0, s1, weights1_up, weights2_up)

        x = self.conv_segmentation(s1)
        if self._use_softmax_head:
            x = self.softmax(x)

        return x


class NasUnetSearch(nn.Module):

    def __init__(self, input_c, c, num_classes, depth, meta_node_num=4,
                 use_sharing=True, double_down_channel=True, use_softmax_head = False, multi_gpus=False, device='cuda'):
        super(NasUnetSearch, self).__init__()
        self._use_sharing = use_sharing
        self._meta_node_num = meta_node_num

        self.net = SearchULikeCNN(input_c, c, num_classes, depth, meta_node_num,
                                  double_down_channel, use_softmax_head)

        if 'cuda' == str(device.type) and multi_gpus:
            device_ids = list(range(torch.cuda.device_count()))
            self.device_ids = device_ids
        else:
            self.device_ids = [0]

        # Initialize architecture parameters: alpha
        self._init_alphas()


    def _init_alphas(self):

        normal_num_ops = len(CellPos)
        down_num_ops = len(CellLinkDownPos)
        up_num_ops = len(CellLinkUpPos)

        k = sum(1 for i in range(self._meta_node_num) for n in range(2 + i))  # total number of input node
        self.alphas_down  = nn.Parameter(1e-3*torch.randn(k, down_num_ops))
        self.alphas_up = nn.Parameter(1e-3*torch.randn(k, up_num_ops))
        self.alphas_normal_down = nn.Parameter(1e-3*torch.randn(k, normal_num_ops))
        self.alphas_normal_up =  self.alphas_normal_down if self._use_sharing else nn.Parameter(1e-3*torch.randn(k, normal_num_ops))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alphas' in n: # TODO: it is a trick, because the parameter name is the prefix of self.alphas_xxx
                self._alphas.append((n, p))

        self._arch_parameters = [
            self.alphas_normal_down,
            self.alphas_down,
            self.alphas_normal_up,
            self.alphas_up
        ]

    def load_alphas(self, alphas_dict):
        self.alphas_down = alphas_dict['alphas_down']
        self.alphas_up = alphas_dict['alphas_up']
        self.alphas_normal_down = alphas_dict['alphas_normal_down']
        self.alphas_normal_up = alphas_dict['alphas_normal_up']
        self._arch_parameters = [
            self.alphas_normal_down,
            self.alphas_down,
            self.alphas_normal_up,
            self.alphas_up
        ]

    def alphas_dict(self):
        return {
            'alphas_down': self.alphas_down,
            'alphas_normal_down': self.alphas_normal_down,
            'alphas_up': self.alphas_up,
            'alphas_normal_up': self.alphas_normal_up,
        }

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        # Note: Since we stack cells by s0: prev prev cells output; s1: prev cells output
        # and when a cell is a up cell, the s0 will be horizontal input and can't do up operation
        # which is different from down cells (s0 and s1 all need down operation). so when
        # parse a up cell string, the string operations is |_|*|_|...|_|, where * indicate up operation
        # mask1 and mask2 below is convenient to handle it.
        geno_parser = GenoParser(self._meta_node_num)
        gene_down = geno_parser.parse(F.softmax(self.alphas_normal_down, dim=-1).detach().cpu().numpy(),
                           F.softmax(self.alphas_down, dim=-1).detach().cpu().numpy(), cell_type='down')
        gene_up = geno_parser.parse(F.softmax(self.alphas_normal_up, dim=-1).detach().cpu().numpy(),
                         F.softmax(self.alphas_up, dim=-1).detach().cpu().numpy(), cell_type='up')

        concat = range(2, self._meta_node_num+2)
        geno_type = Genotype(
            down=gene_down, down_concat = concat,
            up=gene_up, up_concat=concat
        )
        return geno_type

    def forward(self, x):

        weights1_down = F.softmax(self.alphas_normal_down, dim=-1)
        weights1_up = F.softmax(self.alphas_normal_up, dim=-1)
        weights2_down = F.softmax(self.alphas_down, dim=-1)
        weights2_up = F.softmax(self.alphas_up, dim=-1)

        if len(self.device_ids) == 1:
            return self.net(x, weights1_down, weights1_up, weights2_down, weights2_up)

        # scatter x
        xs = nn.parallel.scatter(x, self.device_ids)
        # broadcast weights
        wnormal_down_copies = broadcast_list(weights1_down, self.device_ids)
        wnormal_up_copies = broadcast_list(weights1_up, self.device_ids)
        wdown_copies = broadcast_list(weights2_down, self.device_ids)
        wup_copies = broadcast_list(weights2_up, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas, list(zip(xs, wnormal_down_copies, wnormal_up_copies,
                                                                wdown_copies, wup_copies)),
                                                                devices=self.device_ids)

        return nn.parallel.gather(outputs, self.device_ids[0])

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p

class Architecture(object):

    def __init__(self, model, arch_optimizer, criterion):
        self.model = model
        self.optimizer = arch_optimizer
        self.criterion = criterion

    def step(self, input_valid, target_valid):
        """Do one step of gradient descent for architecture parameters

        Args:
            input_valid: A tensor with N * C * H * W for validation data
            target_valid: A tensor with N * 1 for validation target
            eta:
            network_optimizer:
        """

        self.optimizer.zero_grad()
        logits = self.model(input_valid)
        loss = self.criterion(logits, target_valid)
        loss.backward()
        self.optimizer.step()
