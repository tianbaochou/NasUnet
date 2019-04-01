from torch.functional import F
from util.prim_ops_set import *
from util.utils import consistent_dim
from models.geno_types import Genotype

class MixedOp(nn.Module):

    def __init__(self, c, stride, use_transpose=False):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        if stride >= 2: # down or up edge
            primitives = CellLinkUpPos if use_transpose else CellLinkDownPos
            self._op_type = 'up_or_down'
        else:
            primitives = CellPos
            self._op_type = 'normal'
        for pri in primitives:
            op = OPS[pri](c, stride, affine=False, dp=0)
            self._ops.append(op)

    def forward(self, x, weights1, weights2):
        # weights1: M * 1 where M is the number of normal primitive operations
        # weights1: K * 1 where K is the number of up or down primitive operations
        # Todo: we have three different weights

        if self._op_type == 'up_or_down':
            rst = sum(w * op(x) for w, op in zip(weights2, self._ops))
        else:
            rst = sum(w * op(x) for w, op in zip(weights1, self._ops))
        return rst

class Cell(nn.Module):

    def __init__(self, meta_node_num, c_prev_prev, c_prev, c, cell_type):
        super(Cell, self).__init__()
        self.c_prev_prev = c_prev_prev
        self.c_prev = c_prev
        self.c = c
        self._meta_node_num = meta_node_num
        self._multiplier = meta_node_num
        self._input_node_num = 2

        if cell_type == 'down':
            # Note: the s0 size is twice than s1!
            self.preprocess0 = ConvOps(c_prev_prev, c, kernel_size=1, stride=2, affine=False, ops_order='act_weight_norm')
        else:
            self.preprocess0 = ConvOps(c_prev_prev, c, kernel_size=1, affine=False, ops_order='act_weight_norm')
        self.preprocess1 = ConvOps(c_prev, c, kernel_size=1, affine=False, ops_order='act_weight_norm')

        self._ops = nn.ModuleList()

        # inp2changedim = 2 if cell_type == 'down' else 1
        idx_up_or_down_start = 0 if cell_type == 'down' else 1
        for i in range(self._meta_node_num):
            for j in range(self._input_node_num + i): # the input id for remaining meta-node
                stride = 2 if j < 2 and j >= idx_up_or_down_start else 1 # only the first input is reduction
                # down cell: |_|_|_|_|*|_|_|*|*|_|_|*|*|*| where _ indicate down operation
                # up cell:   |*|_|*|*|_|*|_|*|*|*|_|*|*|*| where _ indicate up operation
                op = MixedOp(c, stride, use_transpose=True) if cell_type=='up' else MixedOp(c, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weight1, weight2):
        # weight1: the normal operations weights with sharing
        # weight2: the down or up operations weight, respectively

        # the cell output is concatenate, so need a convolution to learn best combination
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0

        for i in range(self._meta_node_num):
            # handle the un-consistent dimension
            tmp_list = []
            for j, h in enumerate(states):
                tmp_list += [self._ops[offset+j](h, weight1[offset+j], weight2[offset+j])]
            s = sum(consistent_dim(tmp_list))
            #s = sum(self._ops[offset+j](h, weight1[offset+j], weight2[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)

class NasUnetSearch(nn.Module):

    def __init__(self, input_c, c, num_classes, depth, criterion, meta_node_num=4,
                 use_sharing=True, double_down_channel=True, use_softmax_head = False, use_gpu=True, device='cuda'):
        super(NasUnetSearch, self).__init__()
        self._num_classes =  num_classes   # 2
        self._depth = depth
        self._criterion = criterion
        self._meta_node_num = meta_node_num
        self._multiplier = meta_node_num
        self._use_gpu = use_gpu
        self._use_sharing = use_sharing
        self._device = device
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

        # Initialize architecture parameters: alpha
        self._init_alphas()

    def _init_alphas(self):
        k = sum(1 for i in range(self._meta_node_num) for n in range(2+i)) # total number of input node
        normal_num_ops = len(CellPos)
        down_num_ops = len(CellLinkDownPos)
        up_num_ops = len(CellLinkUpPos)

        # Requires gradient
        if self._use_gpu:
            self.alphas_down = torch.tensor(1e-3*torch.randn(k, down_num_ops).to(self._device), requires_grad=True)
            self.alphas_up = torch.tensor(1e-3*torch.randn(k,up_num_ops).to(self._device), requires_grad=True)
            self.alphas_normal_down = torch.tensor(1e-3*torch.randn(k,normal_num_ops).to(self._device), requires_grad=True)
            self.alphas_normal_up = torch.tensor(1e-3*torch.randn(k,normal_num_ops).to(self._device), requires_grad=True)
        else:
            self.alphas_down = torch.tensor(1e-3*torch.randn(k, down_num_ops), requires_grad=True)
            self.alphas_up = torch.tensor(1e-3*torch.randn(k,up_num_ops), requires_grad=True)
            self.alphas_normal_down = torch.tensor(1e-3*torch.randn(k,normal_num_ops), requires_grad=True)
            self.alphas_normal_up = torch.tensor(1e-3 * torch.randn(k, normal_num_ops), requires_grad=True)


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
        def _parse(weights1, weights2, cell_type):
            gene = []
            n = 2 # indicate the all candidate index for current meta_node
            start = 0
            inp2changedim = 2 if cell_type == 'down' else 1
            nc, no = weights1.shape
            for i in range(self._meta_node_num):

                normal_op_end = start + n
                up_or_down_op_end = start + inp2changedim

                mask1 = np.zeros(nc, dtype=bool)
                mask2 = np.zeros(nc, dtype=bool)

                if cell_type == 'down':
                    mask1[up_or_down_op_end:normal_op_end] = True
                    mask2[start:up_or_down_op_end] = True
                else:
                    mask1[up_or_down_op_end+1:normal_op_end] = True
                    mask1[start:up_or_down_op_end] = True
                    mask2[up_or_down_op_end] = True


                W1 = weights1[mask1].copy()
                W2 = weights2[mask2].copy()
                gene_item1, gene_item2 = [], []
                # Get the k largest strength of mixed up or down edges, which k = 2
                if len(W2) >= 1:
                    edges2 = sorted(range(inp2changedim),
                                    key=lambda x:-max(W2[x][k] for k in range(len(W2[x]))))[:min(len(W2), 2)]

                    # Get the best operation for up or down operation
                    CellPrimitive = CellLinkUpPos if cell_type=='up' else CellLinkDownPos
                    for j in edges2:
                        k_best = None
                        for k in range(len(W2[j])):
                            if k_best is None or W2[j][k] > W2[j][k_best]:
                                k_best = k

                        # Geno item: (weight_value, operation, node idx)
                        gene_item2.append((W2[j][k_best],CellPrimitive[k_best],
                                j if cell_type=='down' else j+1))

                # Get the k largest strength of mixed normal edges, which k = 2
                if len(W1) > 0:
                    edges1 = sorted(range(len(W1)), key=lambda x:-max(W1[x][k]
                                for k in range(len(W1[x])) if k != CellPos.index('none')))[:min(len(W1),2)]
                    # Get the best operation for normal operation
                    for j in edges1:
                        k_best = None
                        for k in range(len(W1[j])):
                            if k != CellPos.index('none'):
                                if k_best is None or W1[j][k] > W1[j][k_best]:
                                    k_best = k

                        # Gene item: (weight_value, operation, node idx)
                        gene_item1.append((W1[j][k_best], CellPos[k_best],
                                           0 if j == 0 and cell_type=='up' else j+inp2changedim))

                # normalize the weights value of gene_item1 and gene_item2
                if len(W1) > 0 and len(W2) > 0 and len(W1[0]) != len(W2[0]):
                    normalize_scale = min(len(W1[0]), len(W2[0])) / max(len(W1[0]), len(W2[0]))
                    if len(W1[0]) > len(W2[0]):
                        gene_item2 = [(w*normalize_scale, po, fid) for (w, po, fid) in gene_item2]
                    else:
                        gene_item1 = [(w*normalize_scale, po, fid) for (w, po, fid) in gene_item1]

                # get the final k=2 best edges
                gene_item1 += gene_item2
                gene += [(po, fid) for (_, po, fid) in sorted(gene_item1)[-2:]]

                start = normal_op_end
                n += 1
            return gene

        gene_down = _parse(F.softmax(self.alphas_normal_down, dim=-1).data.cpu().numpy(),
                           F.softmax(self.alphas_down, dim=-1).data.cpu().numpy(), cell_type='down')
        gene_up = _parse(F.softmax(self.alphas_normal_up, dim=-1).data.cpu().numpy(),
                           F.softmax(self.alphas_up, dim=-1).data.cpu().numpy(), cell_type='up')

        concat = range(2, self._meta_node_num+2)
        geno_type = Genotype(
            down=gene_down, down_concat = concat,
            up=gene_up, up_concat=concat
        )
        return geno_type

    def forward(self, x):
        s0, s1 = self.stem0(x), self.stem1(x)
        down_cs = []

        # the sharing parameters for multi-gpus
        # Todo(zbabby:2018/12/28) we will fork this later
        weights1_down = F.softmax(self.alphas_normal_down, dim=-1)
        weights1_up = F.softmax(self.alphas_normal_up, dim=-1) if self._use_sharing else weights1_down
        weights2_down = F.softmax(self.alphas_down, dim=-1)
        weights2_up = F.softmax(self.alphas_up, dim=-1)

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
