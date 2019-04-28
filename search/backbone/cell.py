from torch.functional import F
from util.prim_ops_set import *
from util.utils import consistent_dim
from util.genotype import CellLinkDownPos, CellLinkUpPos, CellPos

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
