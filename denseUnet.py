"""
Portion of this code is from fastmri(https://github.com/facebookresearch/fastMRI) 
Copyright (c) Facebook, Inc. and its affiliates.
Licensed under the MIT License.
"""

import torch
from torch import nn
from torch.nn import functional as F

class DenseBlock(nn.Module):
    """
    A dense block consisting of 4*(Conv,BN,Act) and 1*Conv layers
    """
    def __init__(self, in_chans, out_chans):
        super(DenseBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU()
        )

        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = x + x1
        x2 = self.conv2(x1)
        x2 = x + x1 + x2
        x3 = self.conv3(x2)
        x3 = x + x1 + x2 + x3
        x4 = self.conv4(x3)
        x4 = x + x1 + x2 + x3 + x4
        return self.conv(x4)


class RRDB_Block(nn.Module):
    """
    A residual-in-residual dense block
    """
    def __init__(self, in_chans, out_chans):
        super(RRDB_Block, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        # there are 8 Dense Blocks in original paper
        self.DB = DenseBlock(in_chans, in_chans)
        

    def forward(self, x, a=0.2):
        dense1_out = self.DB(x) * a
        res_sum = x + dense1_out
        dense2_out = self.DB(res_sum) * a
        res_sum = res_sum + dense2_out
        dense3_out = self.DB(res_sum) * a
        res_sum = res_sum + dense3_out
        return res_sum*a + x

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """
    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super(ConvBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU()
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'
    

class LOUPEUNet(nn.Module):
    """
        PyTorch implementation of a U-Net model.
        This is based on:
            Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
            for biomedical image segmentation. In International Conference on Medical image
            computing and computer-assisted intervention, pages 234–241. Springer, 2015.   
        The model takes a real or complex value image and use a UNet to denoise the image. 
        A residual connection is applied to stablize training.       
    """
    def __init__(self,
                 in_chans,
                 out_chans,
                 chans,
                 num_pool_layers,
                 drop_prob,
                 bi_dir=False,
                 old_recon=False,
                 with_uncertainty=False,
                 num_rrdb=4):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.old_recon = old_recon 
        if old_recon:
            assert 0 
            in_chans = in_chans+1 # add mask dim and old reconstruction dim  

        self.with_uncertainty = with_uncertainty

        if with_uncertainty:
            out_chans = out_chans+1

        self.in_chans = in_chans 
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.num_rrdb = num_rrdb

        # downsample layers
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        

        # backbone RRDB layers
        self.rrdb_blocks = nn.ModuleList([RRDB_Block(ch, ch)])
        for i in range(num_rrdb - 1):
            self.backbone += [RRDB_Block(ch, ch)]
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)


        # Upsampling layers
        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, out_chans, drop_prob)]
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=1, stride=1)

        nn.init.normal_(self.conv2[-1].weight, mean=0, std=0.001) 
        self.conv2[-1].bias.data.fill_(0)

    def forward(self, input, old_recon=None, eps=1e-8):
        # input: NCHW 
        # output: NCHW 

        if self.old_recon:
            assert 0 
            output = torch.cat([input, old_recon], dim=1)
        else:
            output = input 
        
        stack = []
        outputs = [output]
        
        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            outputs.append(output)   # appends on position i+1
            output = F.avg_pool2d(outputs[i], kernel_size=2) + output
            if i > 1:
                output += F.avg_pool2d(outputs[i-1], kernel_size=4)
            if i > 2:
                output += F.avg_pool2d(outputs[i-2], kernel_size=8)

        res_output = output
        for rrdb_layer in self.rrdb_blocks:
            output = rrdb_layer(output)

        output = self.conv(output) + res_output

        # Apply up-sampling layers
        up_outputs = []
        for i, layer in enumerate(self.up_sample_layers):
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)
            up_outputs.append(output)
            if i > 0:
                output = output + nn.Upsample(scale_factor=1, mode='bilinear')(outputs[i-1])
            if i > 1:
                output = output + nn.Upsample(scale_factor=2,  mode='bilinear')(outputs[i-2])
            if i > 2:
                output = output + nn.Upsample(scale_factor=4,  mode='bilinear')(outputs[i-3])
            
        out_conv2 = self.conv2(output)

        img_residual = out_conv2[:, :1]

        if self.with_uncertainty:
            map = out_conv2[:, 1:]
        else:
            map = torch.zeros_like(out_conv2)
    
        if self.old_recon:
            return img_residual + old_recon
        else:
            return img_residual + torch.norm(input, dim=1, keepdim=True)
