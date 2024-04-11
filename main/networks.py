'''
 # @ Author: Yichao Cai
 # @ Create Time: 2024-01-17 14:52:29
 # @ Description: Disentangled Networks
 ''' 
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLPBasic(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim, activation, drop_rate=0.5) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, latent_dim, bias=True)
        self.activation = activation()
        self.dropout = nn.Dropout(p=drop_rate)
        self.projection = self.__zero_initial(nn.Linear(latent_dim, out_dim, bias=False))
        
        self.out_dim = out_dim
        self.down_sample = not (in_dim == out_dim)
        
    def __zero_initial(self, module):
        for p in module.parameters():
            p.detach().zero_()
        return module
    
    def __downsample_nn(self, x):
        bs = x.shape[0]
        x = x.unsqueeze(1)
        x = F.interpolate(x, self.out_dim, mode='nearest')
        return x.view([bs, self.out_dim])
    
    def forward(self, x):
        out = self.activation(x)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.projection(out)
        if self.down_sample:
            x = self.__downsample_nn(x)
        return out + x     
    

class ResMLP(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim, activation, drop_rate=0.5, repeat=0, scale=1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, latent_dim, bias=True)        
        self.activation = activation()
        self.dropout = nn.Dropout(p=drop_rate)
        self.projection = self.__zero_initial(nn.Linear(latent_dim, out_dim, bias=False))
        self.scale = scale
        
        self.repeat = repeat
        if repeat > 0:
            self.latent_linear = nn.Sequential(*[nn.Sequential(*[activation(), nn.Linear(latent_dim, latent_dim, bias=True)])
                                                 for _ in range(repeat)])
        self.out_dim = out_dim
        self.down_sample = not (in_dim == out_dim)
    
    def __zero_initial(self, module):
        for p in module.parameters():
            p.detach().zero_()
        return module
    
    def __downsample_nn(self, x):
        bs = x.shape[0]
        x = x.unsqueeze(1)
        x = F.interpolate(x, self.out_dim, mode='nearest')
        return x.view([bs, self.out_dim])
    
    def forward(self, x):
        out = self.activation(x)
        out = self.dropout(out)
        out = self.linear(out)
        if self.repeat > 0:
            out = self.latent_linear(out)
        out = self.projection(out)
        out = out * self.scale
        if self.down_sample:
            x = self.__downsample_nn(x)
        return out + x        


class ConcatMLP(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim, activation, drop_rate=0.5) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, latent_dim, bias=True)
        self.activation = activation()
        self.dropout = nn.Dropout(p=drop_rate)
        self.projection = self.__zero_initial(nn.Linear(latent_dim, latent_dim, bias=False))
        self.out_dim = out_dim
        
    def __zero_initial(self, module):
        for p in module.parameters():
            p.detach().zero_()
        return module
    
    def forward(self, x):
        out = self.activation(x)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.projection(out)
        out = torch.cat((x, out), dim=-1)
        return out


class DenseMLP(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim, activation) -> None:
        super().__init__()
        self.linear1 = self.__zero_initial(nn.Linear(in_dim, latent_dim, bias=True))
        self.activation = activation()
        self.linear2 = self.__zero_initial(nn.Linear(latent_dim, out_dim, bias=True))
        
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        
    def __zero_initial(self, module):
        for p in module.parameters():
            p.detach().zero_()
        return module
    
    @ staticmethod
    def __downsample_nn(x, tar_dim):
        bs = x.shape[0]
        if x.shape[1] == tar_dim:
            return x
        x = x.unsqueeze(1)
        x = F.interpolate(x, tar_dim, mode='nearest')
        return x.view([bs, tar_dim])
    
    def forward(self, x):
        out1 = self.activation(x)
        out1 = self.linear1(out1)
        out1 = out1 + self.__downsample_nn(x, self.latent_dim)
        out2 = self.activation(out1)
        out2 = self.linear2(out1)
        out2 = out2 + self.__downsample_nn(out1, self.out_dim)
        out = out2 + self.__downsample_nn(x, self.out_dim)
        return out


class UMLP(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim, activation, drop_rate=0.5) -> None:
        super().__init__()
        self.activation = activation()
        self.down = nn.Linear(in_dim, latent_dim, bias=True)
        self.up = nn.Linear(latent_dim, in_dim, bias=True)
        self.projection = self.__zero_initial(nn.Linear(in_dim, out_dim, bias=False))
        self.out_dim = out_dim

    def __zero_initial(self, module):
        for p in module.parameters():
            p.detach().zero_()
        return module

    def __interpolate_nn(self, x):
        bs = x.shape[0]
        if x.shape[1] == self.out_dim:
            return x
        x = x.unsqueeze(1)
        x = F.interpolate(x, self.out_dim, mode='nearest')
        return x.view([bs, self.out_dim])

    def forward(self, x):
        f1 = self.activation(x)
        f1 = self.down(f1)
        f2 = self.activation(f1)
        f2 = self.up(f2)
        out = self.projection(f2) + self.__interpolate_nn(x)
        return out


class InnerModalNet(ResMLP):    
    def __init__(self, in_dim, latent_dim, out_dim, activation=nn.LeakyReLU, drop_rate=0, repeat=0, scale=1) -> None:
        super().__init__(in_dim, latent_dim, out_dim, activation, drop_rate, repeat, scale=scale)


# Img->ψ->Θ; txt->β->Θ
class DisentangledNetwork(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim, activation=nn.LeakyReLU, drop_rate=0.0, 
                 which_network=['beta'], repeat=0, scale=1) -> None:
        super().__init__()
        self.network_psi = InnerModalNet(in_dim=in_dim, latent_dim=latent_dim, out_dim=out_dim,
                                         activation=activation, drop_rate=drop_rate, repeat=repeat,
                                         scale=scale)
        self.network_beta = InnerModalNet(in_dim=in_dim, latent_dim=latent_dim, out_dim=out_dim,
                                          activation=activation, drop_rate=drop_rate, repeat=repeat,
                                          scale=scale)
        self.which_network = which_network[0]
        
    # Different training routes
    def forward_beta(self, t):
        return self.network_beta(t)

    def forward_psi(self, x):
        return self.network_psi(x)

    def forward(self, x):
        if self.which_network == 'beta':
            return self.forward_beta(x)
        elif self.which_network == 'psi':
            return self.forward_psi(x)
        else:
            raise NotImplementedError
        

class NormalMLP(nn.Module):
    def __init__(self, in_dim, latent_dim, out_dim, activation, drop_rate=0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, latent_dim, bias=True)
        self.activation = activation()
        self.dropout = nn.Dropout(p=drop_rate)
        self.projection = nn.Linear(latent_dim, out_dim, bias=False)

    def forward(self, x):
        out = self.activation(x)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.projection(out)
        return out

    # Different training routes
    def forward_beta(self, t):
        return self.forward(t)
    
    def infer_img(self, x):
        return self.forward(x)

    def infer_txt(self, t):
        return self.forward(t)
