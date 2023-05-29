import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class STDataset(Dataset):
    def __init__(self, content_urls, style_urls, transform=None):
        self.content_urls = content_urls
        self.style_urls = style_urls
        self.transform = transform

    def __len__(self):
        return len(self.content_urls)

    def __getitem__(self, idx):
        content_url = self.content_urls[idx]
        style_url = self.style_urls[idx]

        content_image = Image.open(content_url).convert('RGB')
        style_image = Image.open(style_url).convert('RGB')

        if self.transform is not None:
            content_image = self.transform(content_image).unsqueeze(0)
            style_image = self.transform(style_image).unsqueeze(0)

        return {'content_image': content_image,
                'style_image': style_image}


class STDataloader:
    def __init__(self,
                 content_datapath,
                 style_datapath,
                 batch_size,
                 transform,
                 max_train_samples,
                 max_eval_samples,
                 seed):
        self.batch_size = batch_size
        self.transform = transform
        self.seed = seed
        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.content_datapath = content_datapath
        self.style_datapath = style_datapath

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def __call__(self, *args, **kwargs):
        n = self.max_train_samples

        content_datapath_list = os.listdir(self.content_datapath)[:n]

        # Only loading the image url to save on Ram
        content_image_urls = []
        for url in content_datapath_list:
            content_image_urls.append(str("content_images/" + url))

        style_datapath_list = os.listdir(self.style_datapath)[:n]

        style_image_urls = []
        for url in style_datapath_list:
            style_image_urls.append(str("style_images/" + url))

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

