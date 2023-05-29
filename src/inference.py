from generator import Generator


style_img_generator = Generator(num_residual=5)

# Generate the final stylized image
final_output = style_img_generator(content_tensor)

# Convert the tensor back to an image
final_image = transforms.ToPILImage()(final_output.squeeze(0).detach().cpu())

# Save the final stylized image
final_image.save("stylized_image.jpg")