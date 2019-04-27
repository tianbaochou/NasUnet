import sys
import os
import platform
import time
from util import visualize
from util.utils import create_exp_dir
import geno_searched

def main(format='svg'):

    genotype_name = 'NAS_UNET_NEW_V2'
    if len(sys.argv) != 2:
        print('usage:\n python {} ARCH_NAME, Default: NAS_UNET_V2'.format(sys.argv[0]))
    else:
        genotype_name = sys.argv[1]

    store_path = './cell_visualize/' + '/{}'.format(format) + '/{}'.format(genotype_name)
    create_exp_dir(store_path)

    if 'Windows' in platform.platform():
        os.environ['PATH'] += os.pathsep + '../3rd_tools/graphviz-2.38/bin/'
    try:
        genotype = eval('geno_types.{}'.format(genotype_name))
    except AttributeError:
        print('{} is not specified in geno_types.py'.format(genotype_name))
        sys.exit(1)

    visualize.plot(genotype.down, store_path+'/DownC', format=format)
    visualize.plot(genotype.up, store_path+'/UpC',format=format)

if __name__ == '__main__':
    # support {'jpeg', 'png', 'pdf', 'tiff', 'svg', 'bmp'
    # 'tif', 'tiff'}
    main(format = 'png')
