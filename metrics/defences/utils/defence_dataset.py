from torch.utils.data import Dataset
import skimage.io as io
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import os
from pathlib import Path
import pandas as pd

class KoniqAttackedDataset(Dataset):
    """Koniq1k dataset with attacked and source images."""

    def __init__(self, src_dir, dest_dir, mos_path, return_mos=True):
        """
        Arguments:
            src_dir (string): Directory with all source images.
            dest_dir (string): Directory with attacked images. For each attacked image, there should be a source image in src_dir and entry in MOS csv. 
            mos_path (string): Path to csv with moses for all source images.
        """
        self.src_dir = src_dir
        self.attacked_dir = dest_dir
        self.mos_path = mos_path
        if return_mos:
            self.moses = pd.read_csv(mos_path, index_col=False)
        self.transform = torchvision.transforms.ToTensor()
        self.dataset_name = "koniq1kAttacked"
        self.return_mos = return_mos
        self.attacked_filepaths = [str(x) for x in Path(self.attacked_dir).iterdir()]

    def __len__(self):
        return len(self.attacked_filepaths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        attacked_path = self.attacked_filepaths[idx]
        clear_path = os.path.join(self.src_dir, Path(self.attacked_filepaths[idx]).stem + '.jpg')
        attacked_name = Path(attacked_path).name
        attacked_image = self.transform(io.imread(attacked_path))
        clear_image = self.transform(io.imread(clear_path))

        #print(attacked_name)
        
        if self.return_mos:
            mos = self.moses.loc[self.moses['image_name'] == Path(attacked_name).stem + '.jpg'].reset_index(drop=True)['MOS'].values[0]
            
        if self.return_mos:
            return  clear_image, attacked_image, attacked_name, mos
        return  clear_image, attacked_image, attacked_name