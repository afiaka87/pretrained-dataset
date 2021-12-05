import torch
from torchvision.transforms.functional import to_tensor
import glob
from PIL import Image


class EncoderBackbone:
    def __init__(self) -> None:
        super().__init__()
        self.name = 'base'

    def download(self):
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError


class ImageGlobDataset(torch.utils.data.Dataset):
    def __init__(self, image_glob: str, image_size: int):
        self.image_paths = glob.glob(image_glob)
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).resize(
            (self.image_size, self.image_size))
        image = to_tensor(image)
        return image, self.image_paths[idx]
