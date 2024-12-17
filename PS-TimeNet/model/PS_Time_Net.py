import torch
import torch.nn as nn

class Downsample_Module(nn.Module):
    def __init__(self, in_chans = 1, win_size = 7, dsr = 2, embed_dim = 64) -> None:
        super().__init__()
        self.emb = nn.Conv1d(in_chans, embed_dim, kernel_size = win_size, stride = dsr, padding= win_size//2)  # 输入通道，输出通道，kernel_size， stride，padding
        self.bn = nn.BatchNorm1d(embed_dim)
        
    def forward(self, x):
        return self.bn(self.emb(x))
    
class Attention_Module(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 9, stride=1, padding="same", groups=dim)
        self.conv_sequential = nn.Conv1d(dim, dim, 11, stride=1, padding = "same", groups=dim, dilation=5)
        self.conv2 = nn.Conv1d(dim, dim, 1)
        
    def forward(self, x):
        iden = x.clone()
        attn = self.conv2(self.conv_sequential(self.conv1(x)))
        return iden * attn

class SAN_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # cff_hid = cff_hid or dim
        # cff_out = cff_out or dim
        
        self.proj_1 = nn.Conv1d(dim, dim, 1)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act1 = nn.GELU()
        
        self.attn = Attention_Module(dim)
        
        self.proj_2 = nn.Conv1d(dim, dim, 1)
        self.bn2 = nn.BatchNorm1d(dim)      
        
    def forward(self, x):
        iden = x.clone()
        
        x = self.act1(self.bn1(self.proj_1(x)))
        x = self.attn(x)
        x = self.bn2(self.proj_2(x))

        return iden + x

class UPsample_Module(nn.Module):
    def __init__(self, dim, out_dim,  k_size = 7, stride=2, last=False) -> None:
        super().__init__()
        self.last = last   
        if not self.last:         
            self.up = nn.Sequential(nn.ConvTranspose1d(dim, dim, k_size, stride = stride, 
                                                       padding=(k_size - stride + 1)//2, 
                                                       groups = dim, 
                                                       output_padding=(k_size-stride)%2),
                                        nn.Conv1d(dim, out_dim, 1),
                                        nn.BatchNorm1d(out_dim),
                                        nn.GELU()
                                    )
        else:
            self.up = nn.Sequential(nn.ConvTranspose1d(dim, dim, k_size, stride = stride,  
                                                       padding=(k_size - stride + 1)//2, 
                                                       groups = dim, 
                                                       output_padding=(k_size-stride)%2),
                                        nn.Conv1d(dim, out_dim, 1),
                                        nn.Sigmoid()
                                    )
            
    def forward(self, x, cat_x=None):
        x = self.up(x)
        if  cat_x is not None:
            x = torch.cat([x, cat_x], dim=1)
        return x

class LayerNorm1d(nn.Module):
    def __init__(self, dim):
        super(LayerNorm1d, self).__init__()
        self.gamma = nn.Parameter(torch.ones([1,dim,1]))
        self.beta = nn.Parameter(torch.zeros([1,dim,1]))
    
    def forward(self, x):
        mean = x.mean(dim=[1,2],keepdim=True)
        var = x.var(dim=[1,2],keepdim=True)
        x = (x - mean) / (torch.sqrt(var + 1e-8))
        return x * self.gamma + self.beta

class SAN(nn.Module):
    def __init__(self, stage_blocks = [4,4,4,4], downsampling_dims = [12,24,36,48], down_strides = [4,2,2,2], upsampling_dims = [36,24,12,1]) -> None:
        super().__init__()
        
        self.num_stages = len(stage_blocks)
        for i in range(self.num_stages):
            down = Downsample_Module(
                in_chans= 1 if i==0 else downsampling_dims[i-1],
                win_size = 7,
                dsr = down_strides[i],
                embed_dim = downsampling_dims[i])
            blocks = nn.ModuleList(
                [SAN_Block(downsampling_dims[i]) for _ in range(stage_blocks[i])])
            norm = LayerNorm1d(downsampling_dims[i])
            
            setattr(self, f"down_{i+1}", down)
            setattr(self, f"blocks_{i+1}", blocks)
            setattr(self, f"norm_{i+1}", norm)
        
        for task in ['SS', 'd']:
            cur_dim = downsampling_dims[-1]
            for j in range(self.num_stages-1):
                out_dim = upsampling_dims[j]
                cat_dim = downsampling_dims[-j-2]
                setattr(self, f'{task}_up_{j+1}', UPsample_Module(cur_dim, out_dim, stride = down_strides[-j-1], last=False))
                cur_dim = out_dim + cat_dim
            setattr(self, f'{task}_up_{self.num_stages}', UPsample_Module(cur_dim, upsampling_dims[-1],  stride = down_strides[0], last=True))

        self.apply(self.init_weights)

    def init_weights(self,m):
        import math
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv1d,nn.ConvTranspose1d)):
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        
        elif isinstance(m, (nn.Linear)):
            nn.init.normal_(m.weight, std=0.01)
            

    def forward_features(self, x):
        feats = []
        for i in range(self.num_stages):
            win_embed = getattr(self, f"down_{i+1}")
            blocks = getattr(self, f"blocks_{i+1}")
            norm = getattr(self, f"norm_{i+1}")
            x = win_embed(x)
            for block in blocks:
                x = block(x)

            x = norm(x)
            feats.append(x)
        return feats
    
    def forward(self, x):
        feats = self.forward_features(x)
        preds = []
        
        for task in ['SS', 'd']:
            pred = feats[-1]
            for j in range(self.num_stages-1):
                pred = getattr(self, f'{task}_up_{j+1}')(pred, feats[-j-2])

            pred = getattr(self, f'{task}_up_{self.num_stages}')(pred).squeeze(1)
            preds.append(pred)

        return preds
    
def get_model(mode = "deeper"):
    if mode == "mini":
        return SAN(stage_blocks = [2,2,2,2], downsampling_dims = [8,16,32,64], upsampling_dims = [32, 16, 8, 1])
    if mode == "wider":
        return SAN(stage_blocks = [2,2,2,2], downsampling_dims = [16, 32, 64, 128], upsampling_dims = [64, 32, 16, 1])
    if mode == "deeper":
        return SAN(stage_blocks = [4,4,4,4], downsampling_dims = [12, 24, 36, 48], upsampling_dims = [36, 24, 12, 1])
         
def get_loss(loss_weight = [0.9, 0.1]): 

    def Loss_balance(preds, targets, type):

        if type == 1:
            return torch.mean(
                nn.SmoothL1Loss(reduction='none', beta=0.05)(preds, targets) *  torch.where(targets>0, 20.0, 1.0)
                )
        
        if type == 2:
            return nn.SmoothL1Loss(reduction='mean', beta=0.05)(preds, targets)
        
    def loss(preds, targets,  type = [1, 2], loss_weight = loss_weight, loss_fn = Loss_balance):
        return sum([
            loss_weight[i] * loss_fn(preds[i], targets[i], type[i])
            for i in range(len(type)) 
            ])

    return loss
    
if __name__ == "__main__":           
        
    import time
    for mode in ['mini', 'deeper', 'wider']:
        
        model = get_model(mode).cuda()
        model.eval()
        x = torch.randn(1, 1, 8000).cuda()
        a,b = model(x)
        print(a.shape)
        print(b.shape)

