import matplotlib.pyplot as plt
import numpy as np


def plot_image(tensor):
    # Convert the tensor to a numpy array
    images = tensor.numpy()

    # Normalize the pixel values to [0, 1]
    images = np.clip(images, 0, 1)

    # Reshape the tensor to (batch, height, width, channels)
    images = np.transpose(images, (0, 3, 2, 1))

    # Iterate over each image in the batch
    for image in images:
        print(image.shape)
        # Remove the batch dimension if it exists
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)

        # Convert image to RGB if channel order is BGR
        if image.shape[-1] == 3 and np.max(image) > 1:
            image = image[..., ::-1]  # Reverse channel order

        # Plot the image
        plt.imshow(image)
        plt.axis('off')
        plt.show()