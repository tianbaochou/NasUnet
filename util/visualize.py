from graphviz import Digraph

def plot(genotype, filename, format='svg'):
    g = Digraph(
        format=format,
        graph_attr = dict(dpi='800'),
        edge_attr = dict(fontsize='20'),
        node_attr = dict(style='filled', shape='rect', align='center',
                         fontsize='20', height='0.5', width='0.5',
                         penwidth='2'),
        engine='dot'
    )
    g.body.extend(['randkdir=LR'])

    g.node('c_{k-2}', fillcolor='darkseagreen2')
    g.node('c_{k-1}', fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2*i, 2*i+1]:
            op, j = genotype[k]
            if j == 0:
                u = 'c_{k-2}'
            elif j == 1:
                u = 'c_{k-1}'
            else:
                u = str(j-2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor='gray')

    g.node('c_{k}', fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), 'c_{k}', fillcolor='gray')

    g.render(filename, view=True)



