import torch
import numpy as np

# Possibility of working with reflexion padding instead of Zero padding

class ImageTransformationNN(torch.nn.Module):
    
    def __init__(self):
        
        super(ImageTransformationNN, self).__init__()
        
        self.down_sample = DownSampleConv()
        self.res = ResidualNet()
        self.up_sample = UpSampleConv()
        
        
    def forward(self, X):
        #print('started image transformation')
        X = self.down_sample(X) 
        
        #print('downsamlping X done', X.shape)
        X = self.res(X)
              
        #print('X out of residual', X.shape)
        y = self.up_sample(X)
        
        # Print('X Up sampling done')
        return y

class DownSampleConv(torch.nn.Module):
    
    def __init__(self):
        
        super(DownSampleConv, self).__init__()
        
        self.conv2d1 = torch.nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=3)
        self.norm1 = torch.nn.InstanceNorm2d(32, affine=True)

        self.conv2d2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.norm2 = torch.nn.InstanceNorm2d(64, affine=True)

        self.conv2d3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm3 = torch.nn.InstanceNorm2d(128, affine=True)
    
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        
        #print('Started down sampling', X.shape)
              
        """y = self.conv2d1(X)
        y = self.norm1(y)
        y = self.relu(y)
        #print('downsampling, block 1: ', y.shape)
        
        y = self.conv2d2(y)
        y = self.norm2(y)
        y = self.relu(y)
        #print('downsampling, block 2: ', y.shape)
            
        y = self.conv2d3(y)
        y = self.norm3(y)
        y = self.relu(y)"""
        #print('downsampling, block 3: ', y.shape)
        y = self.relu(
                self.norm3(
                    self.conv2d3(
                        self.relu(
                            self.norm2(
                                self.conv2d2(
                                    self.relu(
                                        self.norm1(
                                            self.conv2d1(X)))))))))
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
        self.norm1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()
        
        # MISTAKE HERE
        self.conv2d2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.norm2 = torch.nn.InstanceNorm2d(channels, affine=True)
    
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
        #print('block 4 done')
        
        return y
       

        
class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = torch.nn.Parameter(torch.FloatTensor(dim))
        self.shift = torch.nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
    
class UpSampleConv(torch.nn.Module):
    
    def __init__(self):
        
        super(UpSampleConv, self).__init__()
        
        
        '''# Stride can't be float
        self.conv2d1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.norm1 = torch.nn.InstanceNorm2d(64,affine=True)
        self.relu1 = torch.nn.ReLU()

        self.conv2d2 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1)
        self.norm2 = torch.nn.InstanceNorm2d(32,affine=True)
        self.relu2 = torch.nn.ReLU()

        self.conv2d3 = torch.nn.ConvTranspose2d(32, 3, kernel_size=9, stride=1)
        self.norm3 = torch.nn.InstanceNorm2d(3,affine=True)
        self.tanh = torch.nn.Tanh()'''
        
                
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = InstanceNormalization(64)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = InstanceNormalization(32)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, X):
        '''#print('in upsample conv')
        
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
        print('tanh done', y.shape)'''
        
        #print('x in upsample: ', X.shape)
        y = self.relu(self.in4(self.deconv1(X)))
        #print('y1: ', y.shape)
        y = self.relu(self.in5(self.deconv2(y)))
        #print('y2: ', y.shape)
        y = self.deconv3(y)
        #print('y out: ', y.shape)
        return y