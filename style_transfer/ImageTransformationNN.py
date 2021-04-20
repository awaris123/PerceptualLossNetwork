import torch


class ImageTransformationNN(torch.nn.Module):
    
    def __init__(self):
        
        super(ImageTransformationNN, self).__init__()
        
        self.down_sample = DownSampleConv()
        self.res = ResidualNet()
        self.up_sample = UpSampleConv()
        
        
    def forward(self, X):
        """y = self.up_sample(
                self.res(
                    self.down_sample(
                    X
                )))"""
        X = self.down_sample(X) 
        X = self.res(X)
        y = self.up_sample(X)
        return y

class DownSampleConv(torch.nn.Module):
    
    def __init__(self):
        
        super(DownSampleConv, self).__init__()
        
        self.conv2d1 = torch.nn.Conv2d(3, 32, kernel_size=9, stride=1)
        self.norm1 = torch.nn.InstanceNorm2d(32,affine=True)
        self.relu1 = torch.nn.ReLU()

        self.conv2d2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.norm2 = torch.nn.InstanceNorm2d(64,affine=True)
        self.relu2 = torch.nn.ReLU()

        self.conv2d3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.norm3 = torch.nn.InstanceNorm2d(128,affine=True)
        self.relu3 = torch.nn.ReLU()
    
    def forward(self, X):
        y = self.relu3(
            self.norm3(
            self.conv2d3(
                self.relu2(
                self.norm2(
                self.conv2d2(
                    self.relu1(
                    self.norm1(
                    self.conv2d1(
                    X
                    )))
                )))
            )))
        
        return y


class RBlock(torch.nn.Module):
    # Specification:  http://torch.ch/blog/2016/02/04/resnets.html
    
    def __init__(self, channels:int):
        super(RBlock, self).__init__()
        self.conv2d1 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1)
        self.norm1 = torch.nn.InstanceNorm2d(channels,affine=True)
        self.relu = torch.nn.ReLU()
        self.conv2d2 = torch.nn.Conv2d(3, channels, kernel_size=3, stride=1)
        self.norm2 = torch.nn.InstanceNorm2d(channels,affine=True)
    
    def forward(self, X):
        residual = X
        y_hat = self.norm2(
                self.conv2d2(
                    self.relu(
                    self.norm1(
                    self.conv2d1(
                    X
                    )))
                ))
        y = y_hat + residual
        return y 

class ResidualNet(torch.nn.Module):
    
    def __init__(self):
        
        super(ResidualNet, self).__init__()
        
        self.block1 = RBlock(128)
        self.block2 = RBlock(128)
        self.block3 = RBlock(128)
        self.block4 = RBlock(128)
        self.block5 = RBlock(128)
    
    def forward(self, X):
        y = self.block5(
            self.block4(
            self.block3(
            self.block2(
            self.block1(
            X)))))
    

class UpSampleConv(torch.nn.Module):
    
    def __init__(self):
        
        super(UpSampleConv, self).__init__()
        
        self.conv2d1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=0.5)
        self.norm1 = torch.nn.InstanceNorm2d(64,affine=True)
        self.relu1 = torch.nn.ReLU()

        self.conv2d2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=0.5)
        self.norm2 = torch.nn.InstanceNorm2d(32,affine=True)
        self.relu2 = torch.nn.ReLU()

        self.conv2d3 = torch.nn.ConvTranspose2d(32, 3, kernel_size=9, stride=1)
        self.norm3 = torch.nn.InstanceNorm2d(3,affine=True)
        self.tanh = torch.nn.Tanh()
    
    def forward(self, X):
        y = self.tanh(
            self.norm3(
            self.conv2d3(
                self.relu2(
                self.norm2(
                self.conv2d2(
                    self.relu1(
                    self.norm1(
                    self.conv2d1(
                    X
                    )))
                )))
            )))
        
        return y