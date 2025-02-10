import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock2d(nn.Module):
    def __init__(self,in_channels,expand_channels,*args,**kwargs):
        super(AttentionBlock2d,self).__init__(*args,**kwargs)
        self.w_g=nn.Conv2d(in_channels=in_channels,out_channels=expand_channels,kernel_size=1)
        self.w_x=nn.Conv2d(in_channels=in_channels,out_channels=expand_channels,kernel_size=1)
        self.sa=nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=expand_channels,out_channels=in_channels,kernel_size=1),
            nn.Softmax(),
        )

    def forward(self,g,x):
        xin=self.w_x(x)
        gin=self.w_g(g)
        att=xin+gin
        att=self.sa(att)
        return att*x

class UpConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,padding,strides,*args,**kwargs):
        super(UpConv2d,self).__init__(*args,**kwargs)
        outer_padding=kernel_size-strides
        self.upconv=nn.ConvTranspose2d(in_channels=in_channel,
                                       out_channels=out_channel,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       stride=strides,
                                       output_padding=outer_padding
                                    )

    def forward(self,x):
        return self.upconv(x)

class DoubleConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,padding,*args,**kwargs):
        super(DoubleConv2d,self).__init__(*args,**kwargs)
        self.double_conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.double_conv(x)

class AttentionUnet2d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 network_channels:tuple,
                 kernel_size:tuple,
                 strides:tuple,
                 attention_channels:tuple,
                 padding='same',
                 *args,
                 **kwargs):
        super(AttentionUnet2d,self).__init__(*args,**kwargs)
        self.encoder=nn.ModuleDict()
        self.decoder=nn.ModuleDict()
        self.network_depth=len(network_channels)

        current_dim=in_channels

        for layer,channel in enumerate(network_channels):
            self.encoder[f'enc{layer}']=DoubleConv2d(in_channel=current_dim,out_channel=channel,kernel_size=kernel_size[layer],padding=padding)
            if layer<=len(strides)-1:
                self.encoder[f'pool{layer}']=nn.MaxPool2d(kernel_size=strides[layer],stride=strides[layer])
            current_dim=channel
        
        decoder_length=len(network_channels)-1
        for layer,channel in enumerate(network_channels[:-1][::-1]):
            self.decoder[f'up{layer}']=UpConv2d(in_channel=current_dim,out_channel=channel,kernel_size=kernel_size[decoder_length-1-layer],
                                                padding=1,strides=strides[decoder_length-1-layer])
            self.decoder[f'att{layer}']=AttentionBlock2d(in_channels=channel,expand_channels=attention_channels[decoder_length-1-layer])
            self.decoder[f'dec{layer}']=DoubleConv2d(in_channel=channel*2,out_channel=channel,kernel_size=kernel_size[decoder_length-1-layer],padding=padding)
            current_dim=channel
        
        self.out_layer=nn.Conv2d(in_channels=current_dim,out_channels=out_channels,kernel_size=1)

    def forward(self,x):
        enc_paths=[]
        for layer in range(self.network_depth-1):
            x=self.encoder[f'enc{layer}'](x)
            enc_paths.append(x)
            x=self.encoder[f'pool{layer}'](x)
        x=self.encoder[f'enc{self.network_depth-1}'](x)

        for layer in range(self.network_depth-1):
            up=self.decoder[f'up{layer}'](x)
            att=self.decoder[f'att{layer}'](enc_paths[self.network_depth-2-layer],up)
            con=torch.cat([att,up],axis=1)
            x=self.decoder[f'dec{layer}'](con)
        
        return self.out_layer(x)