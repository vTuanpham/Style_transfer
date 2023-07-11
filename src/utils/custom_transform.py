import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF


class AddGaussianNoise:
    def __init__(self, mean=0., sigma_range=(0., .06), p=0.5):
        self.mean = mean
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            # Convert tensor images to numpy arrays and adjust their shape if needed
            img = img.squeeze().permute(1, 2, 0).detach().cpu().numpy()  # Adjust dimensions as per your tensor shape
            # Normalize 0 - 1
            img = np.interp(img, (img.min(), img.max()), (0, 1))
            h, w, c = img.shape
            sigma = np.random.uniform(*self.sigma_range)
            noise = np.random.normal(self.mean, sigma, (h, w, c))
            noisy_img = img + noise
            noisy_img = np.interp(noisy_img, (noisy_img.min(), noisy_img.max()), (0, 255)).astype(np.uint8)
            noisy_img = TF.to_tensor(noisy_img)
            return noisy_img
        else:
            return img


class RGBToGrayscaleStacked:
    def __init__(self, enable: bool=False):
        self.enable = enable

    def __call__(self, img):
        if self.enable:
            gray_img = torch.zeros_like(img)
            gray_img[0, :, :] = img[0, :, :] * 0.299
            gray_img[1, :, :] = img[1, :, :] * 0.587
            gray_img[2, :, :] = img[2, :, :] * 0.144
            gray_img = torch.sum(gray_img, dim=0, keepdim=True)
            gray_img = torch.cat([gray_img] * 3, dim=0)

            return gray_img
        return img


if __name__ == "__main__":
    # Test the transform on a sample RGB image
    image_path = r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\stylized_image.jpg"
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        AddGaussianNoise(mean=0.5, sigma_range=(0., 0.08), p=0.9),
    ])

    # Convert PIL image to tensor and add noise
    noisy_img_tensor = transform(img).squeeze().permute(1, 2, 0)

    # Plot the noisy image
    plt.imshow(noisy_img_tensor)
    plt.axis("off")
    plt.show()
