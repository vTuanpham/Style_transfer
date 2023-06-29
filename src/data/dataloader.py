import os
import random
import warnings
import math
from typing import List, Dict

import PIL.Image
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


class STDataset(Dataset):
    def __init__(self, content_urls: str, style_urls: str,
                 transform_content=None, transform_style=None,
                 url_only: bool = False):
        self.content_urls = content_urls
        self.style_urls = style_urls
        self.transform_content = transform_content
        self.transform_style = transform_style
        self.url_only = url_only

    def __len__(self) -> int:
        return len(self.content_urls) if len(self.content_urls) < len(self.style_urls) else len(self.style_urls)

    def __getitem__(self, idx) -> Dict:
        content_url = self.content_urls[idx]
        style_url = self.style_urls[idx]

        if not self.url_only:
            try:
                content_image = Image.open(content_url).convert('RGB')
                style_image = Image.open(style_url).convert('RGB')
            except IOError:
                warnings.warn("IO Error, watch out !")
                raise "IO Error!"

            if self.transform_content is not None:
                content_image = self.transform_content(content_image)
            if self.transform_style is not None:
                style_image = self.transform_style(style_image)

            return {'content_image': content_image,
                    'style_image': style_image}
        else:
            return {'content_image': content_url,
                    'style_image': style_url}


class STDataloader:
    def __init__(self,
                 content_datapath: List[str],
                 style_datapath: List[str],
                 eval_contentpath: str,
                 eval_stylepath: str,
                 batch_size: int,
                 transform_content,
                 transform_style,
                 max_style_train_samples: int,
                 max_content_train_samples: int,
                 eval_batch_size: int = 1,
                 seed: int = 42,
                 num_worker: int = 2):
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.transform_content = transform_content
        self.transform_style = transform_style

        self.max_style_train_samples = max_style_train_samples
        self.max_content_train_samples = max_content_train_samples
        self.content_datapath = content_datapath
        self.style_datapath = style_datapath
        self.eval_contentpath = eval_contentpath
        self.eval_stylepath = eval_stylepath

        self.seed = seed
        self.num_worker = num_worker
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    @staticmethod
    def get_img_path_list(dir_path) -> List[str]:
        urls = []
        path_list = os.listdir(dir_path)
        for url in path_list:
            if os.path.isfile(os.path.join(dir_path, url)):
                extension = url.split(".")[-1]
                if extension in ["jpg"] or extension in ["png"]:
                    urls.append(os.path.join(dir_path, url))
                    continue
                warnings.warn("non-image file found in dir.")
            else:
                warnings.warn("Non file found in data dir")
                continue

        return urls

    def __call__(self, *args, **kwargs) -> Dict:
        # Only loading the image url to save on Ram
        content_image_urls = []
        per_max_content_train_samples = math.floor(self.max_content_train_samples / len(self.content_datapath))
        for idx, path in enumerate(self.content_datapath):
            print(f"\n Loading {per_max_content_train_samples} samples in {self.content_datapath[idx].split('/')[-1]}...")
            content_image_urls += self.get_img_path_list(path)[:per_max_content_train_samples]

        style_image_urls = []
        per_max_style_train_samples = math.floor(self.max_style_train_samples / len(self.style_datapath))
        for idx, path in enumerate(self.style_datapath):
            print(f"\n Loading {per_max_style_train_samples} samples in {self.style_datapath[idx].split('/')[-1]}...")
            style_image_urls += self.get_img_path_list(path)[:per_max_style_train_samples]

        content_eval_urls = self.get_img_path_list(self.eval_contentpath)
        style_eval_urls = self.get_img_path_list(self.eval_stylepath)

        train_dataset = STDataset(content_image_urls,style_image_urls,
                                  transform_content=self.transform_content, transform_style=self.transform_style)

        eval_dataset = STDataset(sorted(content_eval_urls), sorted(style_eval_urls), url_only=True)

        return {"train":self.get_dataloader(train_dataset, shuffle_flag=True, batch_size=self.batch_size),
                "eval": self.get_dataloader(eval_dataset, shuffle_flag=False, batch_size=self.eval_batch_size)}

    def get_dataloader(self, dataset, shuffle_flag: bool = False, batch_size: int = 1) -> DataLoader:
        sampler = RandomSampler(data_source=dataset, generator=self.generator) if shuffle_flag else \
            SequentialSampler(dataset)
        return DataLoader(dataset,
                          sampler=sampler,
                          batch_size=batch_size,
                          drop_last=True,
                          num_workers=self.num_worker)


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