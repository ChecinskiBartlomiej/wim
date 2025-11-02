from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CustomCifar10Dataset(Dataset):
    """Reads CIFAR-10 from PNG images"""

    def __init__(self, image_dir, num_datapoints=None):
        """
        Args:
            image_dir: Path to directory containing PNG images
            num_datapoints: Optional limit on number of images to load
        """
        super().__init__()

        self.image_dir = Path(image_dir)

        # Get all PNG files in the directory
        self.image_paths = sorted(list(self.image_dir.glob("*.png")))

        if num_datapoints is not None:
            self.image_paths = self.image_paths[:num_datapoints]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load image as RGB
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')

        # Convert to tensor [0, 1] and normalize to [-1, 1]
        img_tensor = transforms.ToTensor()(img)
        img_tensor = 2 * img_tensor - 1

        return img_tensor
