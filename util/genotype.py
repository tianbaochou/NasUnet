from collections import namedtuple
import numpy as np

Genotype = namedtuple('Genotype', 'down down_concat up up_concat')

CellLinkDownPos = [
    'avg_pool',
    'max_pool',
    'down_cweight',
    'down_dil_conv',
    'down_dep_conv',
    'down_conv'
]

CellLinkUpPos = [
    'up_cweight',
    'up_dep_conv',
    'up_conv',
    'up_dil_conv'
]

CellPos = [
    'identity',
    'none',
    'cweight',
    'dil_conv',
    'dep_conv',
    'shuffle_conv',
    'conv',
]

class GenoParser:
    def __init__(self, meta_node_num=4):
        self._meta_node_num = meta_node_num

    def parse(self, weights1, weights2, cell_type):
        gene = []
        n = 2  # indicate the all candidate index for current meta_node
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
                mask1[up_or_down_op_end + 1:normal_op_end] = True
                mask1[start:up_or_down_op_end] = True
                mask2[up_or_down_op_end] = True

            W1 = weights1[mask1].copy()
            W2 = weights2[mask2].copy()
            gene_item1, gene_item2 = [], []
            # Get the k largest strength of mixed up or down edges, which k = 2
            if len(W2) >= 1:
                edges2 = sorted(range(inp2changedim),
                                key=lambda x: -max(W2[x][k] for k in range(len(W2[x]))))[:min(len(W2), 2)]

                # Get the best operation for up or down operation
                CellPrimitive = CellLinkUpPos if cell_type == 'up' else CellLinkDownPos
                for j in edges2:
                    k_best = None
                    for k in range(len(W2[j])):
                        if k_best is None or W2[j][k] > W2[j][k_best]:
                            k_best = k

                    # Geno item: (weight_value, operation, node idx)
                    gene_item2.append((W2[j][k_best], CellPrimitive[k_best],
                                       j if cell_type == 'down' else j + 1))

            # Get the k largest strength of mixed normal edges, which k = 2
            if len(W1) > 0:
                edges1 = sorted(range(len(W1)), key=lambda x: -max(W1[x][k]
                                                                   for k in range(len(W1[x])) if
                                                                   k != CellPos.index('none')))[:min(len(W1), 2)]
                # Get the best operation for normal operation
                for j in edges1:
                    k_best = None
                    for k in range(len(W1[j])):
                        if k != CellPos.index('none'):
                            if k_best is None or W1[j][k] > W1[j][k_best]:
                                k_best = k

                    # Gene item: (weight_value, operation, node idx)
                    gene_item1.append((W1[j][k_best], CellPos[k_best],
                                       0 if j == 0 and cell_type == 'up' else j + inp2changedim))

            # normalize the weights value of gene_item1 and gene_item2
            if len(W1) > 0 and len(W2) > 0 and len(W1[0]) != len(W2[0]):
                normalize_scale = min(len(W1[0]), len(W2[0])) / max(len(W1[0]), len(W2[0]))
                if len(W1[0]) > len(W2[0]):
                    gene_item2 = [(w * normalize_scale, po, fid) for (w, po, fid) in gene_item2]
                else:
                    gene_item1 = [(w * normalize_scale, po, fid) for (w, po, fid) in gene_item1]

            # get the final k=2 best edges
            gene_item1 += gene_item2
            gene += [(po, fid) for (_, po, fid) in sorted(gene_item1)[-2:]]

            start = normal_op_end
            n += 1
        return gene