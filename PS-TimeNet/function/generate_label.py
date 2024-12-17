import h5py
import numpy as np
import pickle
import torch.utils.data as data
from pathlib import Path

class SS_data_label(data.Dataset):
    
    def __init__(self, data_root, mode:str="train", samples:int=4000, channels:int=1, 
                 label_type:str='triangular', gaussian_sigma = 10) -> None:
        
        # Initialization
        super(SS_data_label, self).__init__()

        self.data_root = Path(data_root)
        self.samples = samples
        self.channels = channels
        self.label_type = label_type
        self.gaussian_sigma = gaussian_sigma

        
        mode_h5 =  h5py.File(self.data_root.joinpath(f'{mode}_SS_data_label.h5'),'r')
        self.data = mode_h5['data']
        self.SS_labels = mode_h5['SS_labels']
        self.p_labels = mode_h5['polarity_labels']
        self.sacname_labels = mode_h5['sacname_labels']
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self._generate_item(index)

    def _generate_item(self, index):
        
        data = self.data[index]
        SSt = self.SS_labels[index]
        polarity = self.p_labels[index]
        sacname = self.sacname_labels[index]

        SS_label, d_label, polarity_label = self._generate_label(SSt, polarity) # 生成标签
        return data, (SS_label, d_label, polarity_label, sacname)

    
    def _generate_label(self, SSt, polarity, label_half_width=80):
        sigma = self.gaussian_sigma
        HW = label_half_width
        d_label = np.zeros((self.samples),dtype=np.float32)
        SS_label = np.zeros((self.samples),dtype=np.float32)
        polarity_label = 0
        # polarity_label[2] = 1 ### positive polarity: 100; negative polarity: 010; without SS：001.
        if SSt == 0:
            return  SS_label, d_label, polarity_label # 执行了return，后面语句就不会再执行了
        
        
        if self.label_type == 'gaussian':
            # SS_label
            if (SSt-HW > 0) and (SSt+HW < self.samples):  
                SS_label[int(SSt-HW):int(SSt+HW)] = np.exp(-(np.arange(-HW,HW))**2/(2*(sigma)**2))
            elif (SSt+HW) >= self.samples:
                SS_label[int(SSt-HW):self.samples] = np.exp(-(np.arange(-HW, self.samples-SSt))**2/(2*(sigma)**2))            
                
        if self.label_type == 'triangular':
            # SS_label
            if (SSt-HW > 0) and (SSt+HW < self.samples):  
                SS_label[SSt-HW:SSt+HW] = 1 - np.abs(np.arange(-HW,HW))/HW
            elif (SSt-HW <= 0):
                SS_label[0:SSt+HW] = 1 - np.abs(np.arange(-SSt,HW))/HW
            elif (SSt+HW) >=self.samples:
                SS_label[SSt-HW:self.samples] = 1 - np.abs(np.arange(-HW, self.samples-SSt))/HW

        # d_label
        if (SSt-1000 > 0) and (SSt+1000 < self.samples):
            d_label[int(SSt-1000):int(SSt+1000+1)] = 1 
        elif (SSt+HW) >= self.samples:
            d_label[int(SSt-1000):self.samples] = 1 
        
        # polarity label
        if polarity == 1:
            polarity_label = 1
            # polarity_label[0] = 1
        elif polarity == -1:
            polarity_label = 2
            # polarity_label[1] = 1
 
        return SS_label, d_label, polarity_label


def get_loader(args, mode, **kwargs):
    
    dataset = SS_data_label(args.data_root, mode, **kwargs)

    return data.DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size,
        shuffle=True if mode=="train" else False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True)


if __name__ == "__main__":

    import argparse
    import os 
    from matplotlib import pyplot as plt
    parser = argparse.ArgumentParser(description="standford")
    parser.add_argument("--batch_size", type= int, default=2)
    parser.add_argument("--num_workers", type = int, default = 0)
    # parser.add_argument("--data_root", type = str, default = "/home/jatq/Documents/STEAD/splits")
    parser.add_argument("--data_root", type = str, default = "D:/_/datasets/STEAD/splits")
    parser.add_argument("--model_name", type = str, default = "san", choices=['san','eqt', 'phasenet','picknet_P', 'picknet_S', 'gpd'])
    args = parser.parse_args("")
    