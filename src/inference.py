import torch
from PIL import Image
from models.generator import Generator
import torchvision.transforms as transforms


style_img_generator = Generator(num_residual=5)
checkpoint = torch.load(r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\models\checkpoints\model.pth")

style_img_generator.load_state_dict(checkpoint)
content_image = Image.open(r'C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\data\mirflickr\im1.jpg').convert('RGB')
trans = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])


content_tensor = trans(content_image)

# Generate the final stylized image
final_output = style_img_generator(content_tensor)

# Convert the tensor back to an image
final_image = transforms.ToPILImage()(final_output.squeeze(0).detach().cpu())

# Save the final stylized image
final_image.save("stylized_image.jpg")