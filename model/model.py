import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import time

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MobileNetV1(BaseModel):
    def __init__(self, get_feature=False, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.get_feature = get_feature
        ##
        print('num_class:{}'.format(num_classes))
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        if self.get_feature:
            return x
        x = self.fc1(x)
        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def speed(model, name):
    t0 = time.time()
    input = torch.rand(1, 3, 224, 224).cuda()
    input = Variable(input, volatile=True)
    t1 = time.time()

    model(input)
    t2 = time.time()

    model(input)
    t3 = time.time()

    print('%10s : %f' % (name, t3 - t2))


if __name__ == '__main__':
    # cudnn.benchmark = True # This will make network slow ??
    resnet18 = models.resnet18().cuda()
    alexnet = models.alexnet().cuda()
    vgg16 = models.vgg16().cuda()
    squeezenet = models.squeezenet1_0().cuda()
    mobilenet = MobileNetV1().cuda()

    speed(resnet18, 'resnet18')
    speed(alexnet, 'alexnet')
    speed(vgg16, 'vgg16')
    speed(squeezenet, 'squeezenet')
    speed(mobilenet, 'mobilenet')
