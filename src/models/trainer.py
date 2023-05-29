import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models
import torchvision.transforms as transforms
import sys
sys.path.insert(0,r'./') #Add root directory here

from losses import style_loss, total_variation_loss, GramMatrix
from losses import mse_loss as content_loss

from generator import Generator
from PIL import Image


# Training loop
for epoch in range(num_epochs):
    # Generate stylized output
    stylized_output = generator(content_tensor)

    # Extract features from VGG19 for content and style images
    content_features = vgg(content_tensor)
    style_features = vgg(style_tensor)
    stylized_features = vgg(stylized_output)

    # Compute content loss
    loss_content = content_loss(stylized_features, content_features)

    # Compute style loss
    loss_style = style_loss(stylized_features, style_features)

    # Compute variation loss
    variation_loss = total_variation_loss(stylized_output)

    # Compute total loss
    total_loss = alpha * loss_content \
                 + beta * loss_style +\
                 gamma * variation_loss

    # Backpropagation and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Print the loss for monitoring
    print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss.item()}")


class Trainer:
    def __init__(self,
                 dataloader,
                 vgg_model_type: str,
                 num_epochs: int,
                 output_path: str,
                 ):

        # Preprocess the images
        self.preprocess = transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.content_img_path = content_img_path
        self.style_img_path = style_img_path
        self.vgg_model_type = vgg_model_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.output_path = output_path

    def get_feature_extractor(self):
        if self.vgg_model_type == '19':
            # Define the VGG19-based feature extractor
            vgg = models.vgg19(pretrained=True).features
            vgg = nn.Sequential(*list(vgg.children())[:36]).eval()
        if self.vgg_model_type == '16':
            # Define the VGG16-based feature extractor
            vgg = models.vgg16(pretrained=True).features
            vgg = nn.Sequential(*list(vgg.children())[:36]).eval()

        return vgg

    def build_model(self):
        generator = Generator(num_residual=5)
        vgg = self.get_feature_extractor()

        return generator, vgg

    def compute_loss(self, content_features, style_features, stylized_features):
        # Compute content loss
        loss_content = content_loss(stylized_features, content_features)
        # Compute style loss
        loss_style = style_loss(stylized_features, style_features)

        # Compute variation loss
        variation_loss = total_variation_loss(stylized_output)

        # Compute total loss
        total_loss = self.alpha * loss_content \
                     + self.beta * loss_style + \
                     self.gamma * variation_loss

        return loss_content, loss_style, total_loss

    def train(self):
        img_generator, vgg = self.build_model()
        # Define the optimizer
        optimizer = optim.Adam(img_generator.parameters(), lr=lr_scheduler)

        # Training loop
        for epoch in range(self.num_epochs):
            img_generator.train()
            for step, batch in enumerate(dataloader):
                content_imgs = batch[:]['content_image']
                style_imgs = batch[:]['style_image']

                # Generate stylized output
                stylized_output = img_generator(content_imgs)

                # Extract features from VGG19 for content and style images
                content_features = vgg(content_imgs)
                style_features = vgg(style_imgs)
                stylized_features = vgg(stylized_output)

                total_loss = self.compute_loss(content_features, style_features, stylized_features)

                # Backpropagation and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            # Print the loss for monitoring
            print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss.item()}")
            self.save(img_generator)

    def save(self, generator):
        torch.save(generator.state_dict(), self.output_path)



