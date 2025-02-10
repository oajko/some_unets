import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Basic 2d Unet:

format:
in_channels int: input channels of
out_channels int: dimension of output mask
channels tuple: channels of encoder and decoder
strides tuple: strides for max_pooling -> singular skip if len(strides)=len(channels)
kernel_size tuple: kernel_size for each channel. Tuple input for each conv inside a layer
padding str or int: Padding option for whole network.

'''


class DoubleConv2d(nn.Module):
    def __init__(self,
                 in_channel:int,
                 out_channel:int,
                 kernel_size:tuple,
                 padding:int,
                 strides:int=1,
                 *args,
                 **kwargs
            ):
        super(DoubleConv2d,self).__init__(*args,**kwargs)

        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)

        self.double_conv=nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size[0],
                padding=padding,
                stride=strides,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=kernel_size[1],
                padding=padding,
                stride=strides,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
    
    def forward(self,x):
        return self.double_conv(x)

class Unet2d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 channels:tuple,
                 pool_strides:tuple=(2,2,2),
                 kernel_sizes:tuple=(3,3,3,3),
                 padding='same',
                 ff_channels:tuple=None,
                 ff_kernel_sizes:tuple=None,
                 *args,
                 **kwargs):
        super(Unet2d,self).__init__(*args,**kwargs)
        self.network_size=len(kernel_sizes)
        self.encoder=nn.ModuleDict()
        self.decoder=nn.ModuleDict()
        self.ff_mod=nn.Sequential()

        current_in=in_channels

        for layer,channel in enumerate(channels):
            self.encoder[f'enc{layer}']=DoubleConv2d(in_channel=current_in,out_channel=channel,kernel_size=kernel_sizes[layer],padding=padding)
            if layer<len(pool_strides):
                self.encoder[f'pool{layer}']=nn.MaxPool2d(kernel_size=pool_strides[layer],stride=pool_strides[layer])
            current_in=channel

        encoder_size=len(kernel_sizes)-1

        for layer,channel in enumerate(channels[:-1][::-1]):
            self.decoder[f'up{layer}']=nn.ConvTranspose2d(in_channels=current_in,out_channels=channel,kernel_size=kernel_sizes[encoder_size-layer],
            padding=1,stride=pool_strides[encoder_size-layer-1],output_padding=kernel_sizes[encoder_size-layer]-pool_strides[encoder_size-layer-1])
            self.decoder[f'dec{layer}']=DoubleConv2d(in_channel=channel*2,out_channel=channel,kernel_size=kernel_sizes[encoder_size-layer],padding=padding)
            current_in=channel
        
        if ff_channels:
            for channel,kernel in zip(ff_channels,ff_kernel_sizes):
                self.ff_mod.add_module(nn.Conv2d(in_channels=current_in,out_channels=channel,kernel_size=kernel))
                current_in=channel

        self.out_conv=nn.Conv2d(in_channels=current_in,out_channels=out_channels,kernel_size=1)

    def forward(self,x):
        enc_out=[]
        for layer in range(self.network_size-1):
            x=self.encoder[f'enc{layer}'](x)
            enc_out.append(x)
            x=self.encoder[f'pool{layer}'](x)
        x=self.encoder[f'enc{self.network_size-1}'](x)

        for layer in range(self.network_size-1):
            tempup=self.decoder[f'up{layer}'](x)
            print([i.shape for i in enc_out])
            print(enc_out[self.network_size-2-layer].shape,tempup.shape)
            x=torch.cat([enc_out[self.network_size-2-layer],tempup],axis=1)
            x=self.decoder[f'dec{layer}'](x)
        
        if len(self.ff_mod)>0:
            x=self.ff_mod(x)
        x=self.out_conv(x)
        return x