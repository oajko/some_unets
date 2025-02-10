import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Make sequence like:
patch size and dim of each patch

selective scan a method of ordering patches. Instead of standard SSM recursive nature, this makes sequence first.

- Diff traversal for patches. Expand dims to b,h,w,c. go into S6 block (SSM mamba). Then we merge back to 1d.

'''

class SSMCalc(nn.Module):
    def __init__(self,l_dim,n_dim,*args,**kwargs):
        super(SSMCalc,self).__init__(*args,**kwargs)
        self.a=nn.Parameter(torch.randn(n_dim,n_dim)*0.1)
        self.b=nn.Parameter(torch.randn(n_dim,l_dim)*0.1)
        self.c=nn.Parameter(torch.randn(l_dim,n_dim)*0.1)
        self.d=nn.Parameter(torch.randn(l_dim,n_dim)*0.1)
    
    def forward(self,x):
        identity=torch.eye(self.a.size(0),device=('cuda' if torch.cuda.is_available() else 'cpu'))

        bx=torch.einsum('ij,bjk->bik',self.b,x)
        h=torch.linalg.solve(identity-self.a,bx)
        h=torch.tanh(torch.einsum('ij,bjk->bik',self.a,h)+bx)
        yhat=torch.einsum('ij,bjk->bik', self.c,h)
        yhat+=self.d
        return yhat

class SSMDiscrete(nn.Module):
    def __init__(self,l_dim,n_dim,*args,**kwargs):
        super(SSMDiscrete,self).__init__(*args,**kwargs)
        self.cross_sections=nn.ModuleList([SSMCalc(l_dim,n_dim) for _ in range(4)])
    
    def forward(self,x):
        y=torch.zeros_like(x)
        b,l,c=x.size()
        h,w=int(l**0.5),int(l**0.5)

        num_=[]
        for i in range(h):
            for j in range(0,l,h):
                if i+j<l:
                    num_.append(i+j)

        y=torch.zeros_like(x)
        patches=self._cross_scan(x)
        for idx,(patch,cross) in enumerate(zip(patches,self.cross_sections)):
            yhat=cross(patch)
            if idx==0:
                y+=yhat
            elif idx==1:
                y+=yhat[:,num_]
            elif idx==2:
                y+=yhat.flip((0,1))
            else:
                y+=yhat[:,num_[::-1]]
        return y/4
    
    def _cross_scan(self,x):
        b,l,d=x.size()
        h,w=int(l**0.5),int(l**0.5)

        num_=[]
        for i in range(h):
            for j in range(0,l,h):
                if i+j<l:
                    num_.append(i+j)     
        vert=x[:,num_]
        hori_rev=x.flip((0,1))
        vert_rev=vert.flip((0,1))
        return x,vert,hori_rev,vert_rev

class VSSBlock(nn.Module):
    '''
    CWCNN is based on seq len - (b,seq_len,seq_dim). Kernel nxnx1, seq_dim has own filter
    '''
    def __init__(self,seq_len,in_dims,expand_dims,out_dims,*args,**kwargs):
        super(VSSBlock,self).__init__(*args,**kwargs)
        self.seq_len=seq_len
        self.vss=nn.Sequential(
            nn.Linear(in_features=in_dims,out_features=expand_dims),
            nn.Sequential(
                nn.Conv1d(in_channels=seq_len,out_channels=seq_len,kernel_size=3,stride=1,padding='same',groups=seq_len),
                nn.SiLU(inplace=True),
            ),
            SSMDiscrete(l_dim=seq_len,n_dim=expand_dims),
            nn.LayerNorm([seq_len,expand_dims]),
        )
        self.layernorm=nn.LayerNorm([seq_len,in_dims])
        self.skiplin=nn.Linear(in_features=in_dims,out_features=expand_dims)
        self.final_lin=nn.Linear(in_features=expand_dims,out_features=out_dims)
    
    def forward(self,x):
        vs=self.layernorm(x)
        skip=self.skiplin(vs)
        vs=self.vss(vs)
        vs=vs*skip
        vs=self.final_lin(vs)
        return vs+x



class VSS(nn.Module):
    def __init__(self,repeats,seq_len,in_dims,expand_dims,out_dims,dec=False,*args,**kwargs):
        super(VSS,self).__init__(*args,**kwargs)
        self.vss_blocks=nn.ModuleList()
        
        for _ in range(repeats):
            self.vss_blocks.append(VSSBlock(seq_len,in_dims,expand_dims,out_dims))
            if dec is True:
                in_dims=in_dims//2

    def forward(self,x):
        for block in self.vss_blocks:
            x=block(x)
        return x

class LinearEmbed(nn.Module):
    '''
    b,c,h,w to b,seq_len,dim
    '''
    def __init__(self,patch_size,embed_dim,*args,**kwargs):
        super(LinearEmbed,self).__init__(*args,**kwargs)
        self.lin=nn.Linear(patch_size**2,embed_dim)
        self.patch_size=patch_size
    
    def forward(self,x):
        b,c,h,w=x.size()
        x=x.reshape(b,c,h//self.patch_size,self.patch_size,w//self.patch_size,self.patch_size)
        x=x.permute(0,2,4,1,3,5)
        x=x.contiguous().view(b,(h//self.patch_size)**2,-1)
        return self.lin(x)

class LinearProject(nn.Module):
    '''
    b,c,h,w to b,seq_len,dim
    '''
    def __init__(self,patch_size,out_dim,*args,**kwargs):
        super(LinearProject,self).__init__(*args,**kwargs)
        self.lin=nn.Linear(patch_size**2,out_dim)
        self.patch_size=patch_size
    
    def forward(self,x):
        b,l,d=x.size()
        x=x.reshape(b,-1,int((l*8)**0.5),int((l*8)**0.5))
        return x
        x=x.permute(0,2,4,1,3,5)
        x=x.contiguous().view(b,(h//self.patch_size)**2,-1)
        return self.lin(x)

class MambaUnet(nn.Module):
    '''
    Input image of B,C,H,W - c=1 (gray-scaled)
    after lin_embed,B,seq_len,dim
    '''
    def __init__(self,init_ps=4,init_embed=16,img_dim=224,*args,**kwargs):
        super(MambaUnet,self).__init__(*args,**kwargs)
        self.lin_embed=LinearEmbed(init_ps,init_embed)
        self.encoder=nn.ModuleDict()
        self.decoder=nn.ModuleDict()

        curr_ps=init_ps
        curr_embed=init_embed
        curr_seq_len=(img_dim//curr_ps)**2
        for layer in range(4):
            self.encoder[f'enc{layer}']=VSS(repeats=2,seq_len=curr_seq_len,in_dims=curr_embed,expand_dims=curr_embed,out_dims=curr_embed)
            if layer!=3:
                curr_ps*=2
                curr_embed*=2
                curr_seq_len=curr_seq_len//2

        for layer in range(3):
            curr_ps=curr_ps//2
            curr_seq_len=curr_seq_len*2
            self.decoder[f'dec{layer}']=VSS(repeats=2,seq_len=curr_seq_len,in_dims=curr_embed,expand_dims=curr_embed,out_dims=curr_embed,dec=False)
            self.decoder[f'lin{layer}']=nn.Linear(curr_embed,curr_embed//2)
            curr_embed=curr_embed//2

        self.lin_project=LinearProject(curr_ps,curr_embed)

    def forward(self,x):
        x=self.lin_embed(x)
        enc_layers=[]
        for layer in range(4):
            x=self.encoder[f'enc{layer}'](x)
            enc_layers.append(x.detach())
            if layer!=3:
                x=self._patch_merge(x,2)
        
        for layer in range(3):
            up=self._patch_expanding(x,2)
            up=torch.cat([enc_layers[2-layer],up],axis=2)
            x=self.decoder[f'dec{layer}'](up)
            x=self.decoder[f'lin{layer}'](x)
        x=self._patch_expanding(x,2)
        x=self.lin_project(x)
        return x
    
    def _patch_merge(self,x,div_factor):
        b,l,d=x.size()
        x=torch.reshape(x,(b,l//div_factor,div_factor,d))
        return torch.flatten(x,start_dim=2)
    
    def _patch_expanding(self,x,div_factor):
        b,l,d=x.size()
        x=torch.reshape(x,(b,l,div_factor,d//div_factor))
        x=torch.flatten(x,start_dim=1,end_dim=2)
        return x

# torch.manual_seed(20)
# model=MambaUnet(4,64,128)
# x=model(torch.randn(1,1,128,128))