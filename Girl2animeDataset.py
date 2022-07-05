from torch.utils.data import Dataset
from PIL import Image
import glob
from pathlib import Path
import numpy as np
class Girl2animeDataset(Dataset):
    def __init__(self, root, transform, mode):
        self.transform = transform
        self.files_A = sorted(list(Path(root, f"{mode}A").glob('*.png')))
        self.files_B = sorted(list(Path(root, f"{mode}B").glob('*.png')))

    def __getitem__(self, index):
        tensor_A = self.transform(Image.open(self.files_A[index]))
        tensor_B = self.transform(Image.open(self.files_B[index]))

        return tensor_A, tensor_B

    def __len__(self):
        return len(self.files_A)