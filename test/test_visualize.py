import test
import time
import platform
from util.utils import *
from util.visualize import Visualize
from models import  get_segmentation_model


class TestVisualize(test.TestCase):
    def setUp(self):
        pass

    def test_op0(self):
        inputs = torch.randn(1, 3, 224, 224)
        store_path = '../experiment/arch_visualize/' + 'arch-{}-{}'.format('exp', time.strftime('%Y%m%d-%H%M'))

        create_exp_dir(store_path)
        model = get_segmentation_model('unet',
                                       dataset='pascal_voc',
                                       backbone=None,
                                       aux=False,
                                       c=32,
                                       depth=4,
                                       genotype=None
                                       )

        y = model(inputs.clone().detach().requires_grad_(True))

        if 'Windows' in platform.platform():
            os.environ['PATH'] += os.pathsep + '../3rd_tools/graphviz-2.38/bin/'

        Vis = Visualize(arch=(model.state_dict(), y), store_path=store_path+'/arch')
        Vis.plot_arch_graph()

    def test_op1(self):
        genotype_name = 'NAS_UNET_V1'
        if len(sys.argv) != 2:
            print('usage:\n python {} ARCH_NAME, Default: NAS_UNET_V1'.format(sys.argv[0]))
        else:
            genotype_name = sys.argv[1]

        store_path = '../experiment/cell_visualize/' + 'cell-{}-{}'.format('exp', time.strftime('%Y%m%d-%H%M'))
        create_exp_dir(store_path)

        if 'Windows' in platform.platform():
            os.environ['PATH'] += os.pathsep + '../3rd_tools/graphviz-2.38/bin/'
        try:
            genotype = eval('geno_types.{}'.format(genotype_name))
        except AttributeError:
            print('{} is not specified in geno_types.py'.format(genotype_name))
            sys.exit(1)

        vis_down = Visualize(genotype.down, store_path=store_path + '/down')
        vis_up = Visualize(genotype.up, store_path=store_path + '/up')
        vis_down.plot_cell()
        vis_up.plot_cell()


if __name__ == '__main__':
    test.main()