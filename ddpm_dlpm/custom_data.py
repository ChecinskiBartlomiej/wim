import pandas as pd
import numpy as np
import torchvision
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms


class CustomMnistDataset(Dataset):
    """Reads MNIST from csv"""

    def __init__(self, csv_path, num_datapoints=None, use_horizontal_flip=False):

        super().__init__()

        self.df = pd.read_csv(csv_path)

        if num_datapoints is not None:
            self.df = self.df.iloc[0:num_datapoints]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img = self.df.iloc[index].filter(regex="pixel").values
        img = np.reshape(img, (28, 28)).astype(np.uint8)

        img_tensor = torchvision.transforms.ToTensor()(img)
        img_tensor = 2 * img_tensor - 1

        return img_tensor

class CustomCifar10Dataset(Dataset):
    """Reads CIFAR-10 from PNG images"""

    def __init__(self, image_dir, num_datapoints=None, use_horizontal_flip=False):
        """
        Args:
            image_dir: Path to directory containing PNG images
            num_datapoints: Optional limit on number of images to load
            use_horizontal_flip: If True, apply random horizontal flips (as in DDPM paper)
        """
        super().__init__()

        self.image_dir = Path(image_dir)
        self.use_horizontal_flip = use_horizontal_flip

        self.image_paths = sorted(list(self.image_dir.glob("*.png")))

        if num_datapoints is not None:
            self.image_paths = self.image_paths[:num_datapoints]

        if self.use_horizontal_flip:
            self.flip_transform = transforms.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')

        if self.use_horizontal_flip:
            img = self.flip_transform(img)

        img_tensor = transforms.ToTensor()(img)
        img_tensor = 2 * img_tensor - 1

        return img_tensor