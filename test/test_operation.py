import unittest
import time
from util.prim_ops_set import *


class TestOperation(unittest.TestCase):
    def setUp(self):
        self.input = torch.randn(40, 32, 16, 16).cuda()
        self.target = torch.randint(0, 10, (40, ), dtype=torch.long).cuda()
        self.fc1 = torch.nn.Linear(32, 10).cuda()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1).cuda()
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.stride = 2
        self.c = 32

    def classifier(self, layer, ):
        out = layer(self.input)
        print('size of feature map: ', out.size())
        out = self.avgpool(out)
        out = self.fc1(out.view(out.size(0), -1))
        loss = self.criterion(out, self.target)
        loss.backward()
        return loss.item()

    def test_op0(self):
        start = time.time()
        layer = ConvOps(self.c , self.c, kernel_size=3).cuda()
        loss = self.classifier(layer=layer)
        end = time.time()
        print('{}x{}_Conv: loss: {}, cost: {}s'.format(3, 3, loss, end-start))

    def test_op1(self):
        start = time.time()
        layer1 = ConvOps(self.c, self.c, dilation=3, stride=2).cuda()
        #layer2 = CWeightOp(self.c, self.c, stride=2, use_transpose=True).cuda()
        layer2 = ConvOps(self.c, self.c, stride=2, dilation=1, use_transpose=True).cuda()
        #layer2 = ConvOps(self.c, self.c, stride=2, use_depthwise=True, use_transpose=True).cuda()
        #layer2 = ConvOps(self.c, self.c, stride=2, use_transpose=True).cuda()
        down = layer1(self.input)
        print('down size: ', down.size())
        up = layer2(self.input)
        print('up size: ', up.size())
        print('up size: ', up.size())
        # loss = self.classifier(layer=layer2)
        end = time.time()
        # print('{}x{}_Conv_TranConv: loss: {}, cost: {}s'.format(3, 3, loss, end-start))

    def test_op2(self):
        start = time.time()
        layer = ConvOps(self.c, self.c, kernel_size=3, stride=2).cuda()
        loss = self.classifier(layer=layer)
        end = time.time()
        print('{}x{}_stride_{}_Conv: loss {}, cost: {}s'.format(3, 3, 2, loss, end-start))

    def test_op3(self):
        start = time.time()
        layer = ConvOps(self.c, self.c, kernel_size=3, dilation=2).cuda()
        loss = self.classifier(layer=layer)
        end = time.time()
        print('{}x{}_Dil_r_{}_Conv: loss {}, cost: {}s'.format(3, 3, 2, loss, end-start))

    def test_ops4(self):
        start = time.time()
        layer = ConvOps(self.c, self.c, kernel_size=3, groups=self.c).cuda()
        loss = self.classifier(layer=layer)
        end = time.time()
        print('{}x{}_Group_Conv: loss {}, cost: {}s'.format(3, 3, loss, end-start))

    def test_op5(self):
        start = time.time()
        layer = IdentityOp(self.c, self.c)
        loss = self.classifier(layer=layer)
        end = time.time()
        print('Identity: loss {}, cost: {}s'.format(loss, end-start))

    def test_op6(self):
        start = time.time()
        layer = CWeightOp(self.c, self.c, kernel_size=3, stride=1).cuda()
        loss = self.classifier(layer=layer)
        end = time.time()
        print('Channel Weight: loss {}, cost: {}s'.format(loss, end-start))

    def test_op7(self):
        start = time.time()
        layer = CWeightOp(self.c, self.c, kernel_size=3, use_transpose=True, stride=2).cuda()
        loss = self.classifier(layer=layer)
        end = time.time()
        print('Tran Channel Weight: loss{}, cost: {}s'.format(loss, end-start))

if __name__ == '__main__':
    test.main()