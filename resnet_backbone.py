import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Input factor should be of 32

Standard decoder

'''

class Resnet2dBlock(nn.Module):
    def __init__(self,in_channels,out_channels,repeats,*args,**kwargs):
        super(Resnet2dBlock,self).__init__(*args,**kwargs)
        in_dim=in_channels
        self.res_blocks=nn.ModuleList()
        for _ in range(repeats):
            self.res_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels=in_dim,out_channels=in_channels,kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1),
            ))
            self.res_blocks.append(nn.ReLU())
    
    def forwrd(self,x):
        for block in range(len(self.res_blocks)//2):
            identity=x
            x=self.res_blocks[block](x)
            x+=identity
            x=self.res_blocks[block+1](x)
        return x

class DoubleConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,*args,**kwargs):
        super(DoubleConv2d,self).__init__(*args,**kwargs)
        self.doubleconv=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self,x):
        return self.doubleconv(x)

class Resnet2dBlock(nn.Module):
    def __init__(self,in_channels,inter_channels,out_channels,repeats,*args,**kwargs):
        super(Resnet2dBlock,self).__init__(*args,**kwargs)
        in_dim=in_channels
        self.res_blocks=nn.ModuleList()
        for _ in range(repeats):
            self.res_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels=in_dim,out_channels=inter_channels,kernel_size=1,padding='same'),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=inter_channels,out_channels=inter_channels,kernel_size=3,padding='same'),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=inter_channels,out_channels=out_channels,kernel_size=1,padding='same'),
            ))
            self.res_blocks.append(nn.ReLU())
            in_dim=out_channels

        self.init_skip=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self,x):
        init_up=True
        for block in range(len(self.res_blocks)//2):
            identity=x
            if init_up is True:
                identity=self.init_skip(x)
                init_up=False
            x=self.res_blocks[block*2](x)
            x+=identity
            x=self.res_blocks[block*2+1](x)
        return x

class DoubleConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,*args,**kwargs):
        super(DoubleConv2d,self).__init__(*args,**kwargs)
        self.doubleconv=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1,padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self,x):
        return self.doubleconv(x)

class ResNet50BackBone2d(nn.Module):
    def __init__(self,in_dims,out_dims,first_channels=64,*args,**kwargs):
        super(ResNet50BackBone2d,self).__init__(*args,**kwargs)

        self.input_conv=nn.Conv2d(in_dims,first_channels,kernel_size=7,stride=2,padding=3)
        self.encoder=nn.ModuleDict()
        self.decoder=nn.ModuleDict()

        for idx,layer in enumerate([3,4,6,3]):
            if idx==0:
                self.encoder[f'enc{idx}']=Resnet2dBlock(first_channels,first_channels,first_channels*4,layer)
            else:
                self.encoder[f'enc{idx}']=Resnet2dBlock(first_channels*2,first_channels,first_channels*4,layer)
            self.encoder[f'pool{idx}']=nn.MaxPool2d(2)
            first_channels*=2

        current_in=first_channels*2

        for layer in range(3):
            self.decoder[f'up{layer}']=nn.ConvTranspose2d(in_channels=current_in,out_channels=current_in//2,kernel_size=3,stride=2,padding=1,output_padding=1)
            self.decoder[f'dec{layer}']=DoubleConv2d(in_channels=current_in,out_channels=current_in//2)
            current_in=current_in//2
        
        self.out_layer=nn.Conv2d(in_channels=current_in,out_channels=out_dims,kernel_size=1)

    def forward(self,x):
        encoder_blocks=[]
        x=self.input_conv(x)
        encoder_blocks.append(x)

        for layer in range(3):
            x=self.encoder[f'enc{layer}'](x)
            encoder_blocks.append(x)
            x=self.encoder[f'pool{layer}'](x)
        x=self.encoder[f'enc{3}'](x)

        for layer in range(3):
            up=self.decoder[f'up{layer}'](x)
            up=torch.cat([encoder_blocks[3-layer],up],axis=1)
            x=self.decoder[f'dec{layer}'](up)
        
        return self.out_layer(x)