# Pretrauined VGG16
# https://pytorch.org/vision/stable/models.html
# https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html#vgg16_bn

import torch
from torchvision.models import vgg16

class VGG16LossNN(torch.nn.Module):
    
    '''
    VGG16 Feature Layers
    
    0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    1 ReLU(inplace=True)
    2 Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    3 ReLU(inplace=True) (RELU1_2)
    
    4 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    5 Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    6 ReLU(inplace=True)
    7 Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    8 ReLU(inplace=True) (RELU2_2)
    
    9 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    10 Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    11 ReLU(inplace=True)
    12 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    13 ReLU(inplace=True)
    14 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    15 ReLU(inplace=True) (RELU3_3)
    
    16 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    17 Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    18 ReLU(inplace=True)
    19 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    20 ReLU(inplace=True)
    21 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    22 ReLU(inplace=True) (RELU4_3)
    
    23 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    24 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    25 ReLU(inplace=True)
    26 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    27 ReLU(inplace=True)
    28 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    29 ReLU(inplace=True)
    30 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    '''
    
    def __init__(self):
        super(VGG16LossNN, self).__init__()
        
        vgg16_features = vgg16(pretrained=True).features
        
        self.section1 = torch.nn.Sequential()
        self.section2 = torch.nn.Sequential()
        self.section3 = torch.nn.Sequential()
        self.section4 = torch.nn.Sequential()
        
        for i, module in enumerate(vgg16_features[0:4]):
            self.section1.add_module(str(i), module)
    
        for i, module in enumerate(vgg16_features[4:9]):
            self.section2.add_module(str(i+4), module)

        for i, module in enumerate(vgg16_features[9:16]):
            self.section3.add_module(str(i+9), module)

        for i, module in enumerate(vgg16_features[16:23]):
            self.section4.add_module(str(i+16), module)
        
        for param in self.parameters():
            param.requires_grad=False
        
        
    def forward(self, X:torch.Tensor) -> dict:
        relu1_2 = self.section1(X)
        relu2_2 = self.section2(relu1_2)
        relu3_3 = self.section3(relu2_2)
        relu4_3 = self.section4(relu3_3)
        return {
            "relu1_2":relu1_2,
            "relu2_2":relu2_2,
            "relu3_3":relu3_3,
            "relu4_3":relu4_3,
            
        }
