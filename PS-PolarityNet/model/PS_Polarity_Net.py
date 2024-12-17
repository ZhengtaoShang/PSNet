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


class seperate_polarities(nn.Module):
    def __init__(self, dim, out_dim, embed_dim=[128, 256]):
        super().__init__()
        self.seperate = nn.Sequential(nn.Conv1d(dim, embed_dim[0], kernel_size = 7, stride = 2, padding= 7//2),
                                      nn.BatchNorm1d(embed_dim[0]),
                                      nn.GELU(),
                                      nn.Conv1d(embed_dim[0], embed_dim[1], kernel_size = 7, stride = 5, padding= 7//2),
                                      nn.BatchNorm1d(embed_dim[1]),
                                      nn.GELU(),
                                      nn.AdaptiveAvgPool1d(1),  # b , 96, 1
                                      nn.Flatten(),              # b , 96
                                      nn.Linear(embed_dim[1], embed_dim[1]),
                                      nn.BatchNorm1d(embed_dim[1]),
                                      nn.GELU(),
                                      nn.Linear(embed_dim[1], embed_dim[1]),
                                      nn.BatchNorm1d(embed_dim[1]),
                                      nn.GELU(),
                                      nn.Linear(embed_dim[1], out_dim),
                                      )

    def forward(self, x):
        x = self.seperate(x)

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
    def __init__(self, stage_blocks = [4,4,4,4], downsampling_dims = [12,24,36,48], down_strides = [4,2,2,2]) -> None:
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
        
        for task in ['polarity']:
            cur_dim = downsampling_dims[-1]
            sep = seperate_polarities(cur_dim, 3)
            setattr(self, f'{task}', sep)

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
        
        for task in ['polarity']:
            pred = getattr(self, f'{task}')(feats[-1])
            preds.append(pred)

        return preds
    
def get_model(mode = 'deeper'):
    if mode == 'mini':
        return SAN(stage_blocks = [3,3,3,3], downsampling_dims = [8, 16, 32, 64])
    if mode == 'wider':
        return SAN(stage_blocks = [2,2,2,2], downsampling_dims = [16, 32, 64, 128])
    if mode == 'deeper':
        # return SAN(stage_blocks = [4,4,4,4], downsampling_dims = [12, 24, 36, 48])
        return SAN(stage_blocks = [4,4,4,4], downsampling_dims = [8, 16, 32, 64])
         
def get_loss():

    def Loss_balance(preds, targets):
        return nn.CrossEntropyLoss(reduction='mean')(preds, targets)
        
    def loss(preds, targets, loss_fn = Loss_balance):
        return loss_fn(preds, targets)

    return loss
    
if __name__ == "__main__":           
        
    import time
    for mode in ['mini', 'deeper', 'wider']:
        
        model = get_model(mode).cuda()
        model.eval()
        x = torch.randn(1, 1, 8000).cuda()
        [a] = model(x)
        print(a.shape)

        loss = get_loss()
        print(loss)





