import os
import random
import warnings
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, RandomSampler


class STDataset(Dataset):
    def __init__(self, content_urls, style_urls, transform=None):
        self.content_urls = content_urls
        self.style_urls = style_urls
        self.transform = transform

    def __len__(self):
        return len(self.content_urls)

    def __getitem__(self, idx):
        content_url = self.content_urls[idx]
        # Randomly choose style for the image
        # style_url = random.choice(self.style_urls)
        style_url = r'C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\data\style_data\Data\Artworks\888440.jpg'
        # style_url = self.style_urls[idx]

        content_image = Image.open(content_url).convert('RGB')
        style_image = Image.open(style_url).convert('RGB')

        if self.transform is not None:
            content_image = self.transform(content_image)
            style_image = self.transform(style_image)

        return {'content_image': content_image,
                'style_image': style_image}


class STDataloader:
    def __init__(self,
                 content_datapath,
                 style_datapath,
                 batch_size,
                 transform,
                 max_style_train_samples,
                 max_content_train_samples,
                 max_eval_samples,
                 seed):
        self.batch_size = batch_size
        self.transform = transform
        self.seed = seed
        self.max_style_train_samples = max_style_train_samples
        self.max_content_train_samples = max_content_train_samples
        self.max_eval_samples = max_eval_samples
        self.content_datapath = content_datapath
        self.style_datapath = style_datapath

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def __call__(self, *args, **kwargs):
        # Only loading the image url to save on Ram

        content_datapath_list = os.listdir(self.content_datapath)[:self.max_content_train_samples]
        content_image_urls = []
        for url in content_datapath_list:
            if os.path.isfile(os.path.join(self.content_datapath, url)):
                extension = url.split(".")[-1]
                assert extension in ["jpg"] or extension in ["png"], "non-image file found in dir."
                content_image_urls.append(os.path.join(self.content_datapath, url))
            else:
                warnings.warn("Non file found in data dir")
                continue

        style_datapath_list = os.listdir(self.style_datapath)[:self.max_style_train_samples]
        style_image_urls = []
        for url in style_datapath_list:
            if os.path.isfile(os.path.join(self.style_datapath, url)):
                extension = url.split(".")[-1]
                assert extension in ["jpg"] or extension in ["png"], "non-image file found in dir."
                style_image_urls.append(os.path.join(self.style_datapath, url))
            else:
                warnings.warn("Non file found in data dir")
                continue

        dataset = STDataset(content_image_urls, style_image_urls, transform=self.transform)
        return self.get_dataloader(dataset, shuffle_flag=True)

    def get_dataloader(self, dataset, shuffle_flag: bool = False):

        sampler = RandomSampler(data_source=dataset, generator=self.generator) if shuffle_flag else \
            SequentialSampler(dataset)
        return DataLoader(dataset,
                          sampler=sampler,
                          batch_size=self.batch_size,
                          drop_last=True,
                          num_workers=2)


if __name__ == "__main__":
    dataloader_args = {
        "content_datapath": '/kaggle/input/images-for-style-transfer/Data/TestCases',
        "style_datapath": '/kaggle/input/images-for-style-transfer/Data/Artworks',
        "batch_size": 3,
        "transform": transforms.Compose([
            transforms.Resize((30, 30)),
            transforms.ToTensor()
        ]),
        "max_train_samples": 10,
        "max_eval_samples": 100,
        "seed": 42
    }
    dataloaders = STDataloader(**dataloader_args)
    dataloaders = dataloaders.__call__()

    for data in iter(dataloaders):
        print(data['style_image'].shape)
        plot_image(data['style_image'])