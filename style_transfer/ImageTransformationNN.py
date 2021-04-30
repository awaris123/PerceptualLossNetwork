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
        #print('started image transformation')
        X = self.down_sample(X) 
        
        #print('downsamlping X done')
        X = self.res(X)
              
        #print('X out of residual')
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

    
# Just added reflection padding 
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
    
    

class RBlock(torch.nn.Module):
    # Specification:  http://torch.ch/blog/2016/02/04/resnets.html
    
    def __init__(self, channels:int):
        super(RBlock, self).__init__()
        self.conv2d1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.norm1 = torch.nn.InstanceNorm2d(channels,affine=True)
        self.relu = torch.nn.ReLU()
        
        # MISTAKE HERE
        self.conv2d2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.norm2 = torch.nn.InstanceNorm2d(channels,affine=True)
    
    def forward(self, X):
        #print('IN RBlock')
        residual = X
        
        y_hat = self.conv2d1(X)
        #print('conv2d1 passed')
        
        y_hat = self.norm1(y_hat)
        #print('norm1 passed')
        
        y_hat = self.relu(y_hat)
        #print('relu passed')
        
        #print('y_hat size: ', y_hat.shape)
        y_hat = self.conv2d2(y_hat)
        #print('conv2d2 passed') 
        
        y_hat = self.norm2(y_hat)
        #print('norm2 passed')
        
        #print('Adding:', y_hat.shape, residual.shape)
        y = y_hat + residual
        #print('addition passed')
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
        #print('In Residual Net')
        
        y = self.block1(X)
        #print('block 1 done')
       
        y = self.block2(y)
        #print('block 2 done')
        
        y = self.block3(y)
        #print('block 3 done')
        
        y = self.block4(y)
        #print('block 4 done')
        
        y = self.block5(y)
        print('block 4 done')
        
        return y
       

class UpSampleConv(torch.nn.Module):
    
    def __init__(self):
        
        super(UpSampleConv, self).__init__()
        
        
        # Stride can't be float
        self.conv2d1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.norm1 = torch.nn.InstanceNorm2d(64,affine=True)
        self.relu1 = torch.nn.ReLU()

        self.conv2d2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1)
        self.norm2 = torch.nn.InstanceNorm2d(32,affine=True)
        self.relu2 = torch.nn.ReLU()

        self.conv2d3 = torch.nn.ConvTranspose2d(32, 3, kernel_size=9, stride=1)
        self.norm3 = torch.nn.InstanceNorm2d(3,affine=True)
        self.tanh = torch.nn.Tanh()
    
    def forward(self, X):
        #print('in upsample conv')
        
        y = self.conv2d1(X)
        print('conv2d1 done', y.shape)
        
        y = self.norm1(y)
        print('norm1 done', y.shape)

        y = self.relu1(y)
        print('relu done', y.shape)

        y = self.conv2d2(y)
        print('conv2d2 done', y.shape)

        y = self.norm2(y)
        print('norm2 done', y.shape)

        y = self.relu2(y)
        print('relu2 done', y.shape)

        y = self.conv2d3(y)
        print('conv2d3 done', y.shape)

        y = self.norm3(y)
        print('norm3 done', y.shape)

        y = self.tanh(y)
        print('tanh done', y.shape)

        
        return y