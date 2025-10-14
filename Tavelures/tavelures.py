import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load the image
folder_path = '0033.jpg'
image_path = folder_path + ''
image = cv2.imread(image_path)


# Convert the image to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#display the gray scale image
plt.figure()
plt.imshow(grayscale_image,cmap = 'gray')
plt.title('grayscale image')
plt.show()

# calculating contrast of the image

average = np.mean(grayscale_image)
standardDeviation = np.std(grayscale_image)
contrast = standardDeviation / average
print('the contrast is :', contrast)

# Compute the histogram
histogram = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.plot(histogram, color='black')
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

def calculate_contrast(window):
    """
    Calculate the contrast of a given window.
    Contrast is defined as the standard deviation of pixel intensities.
    """
    return np.std(window)/np.mean(window)

def moving_window_contrast(image, window_size):
    """
    Compute an image of contrast using a moving window approach.

    :param image: Input grayscale image.
    :param window_size: Size of the moving window (assumed to be square).
    :return: Image of contrast.
    """
    height, width = image.shape
    contrast_image = np.zeros((height, width), dtype=np.float32)
    half_window = window_size // 2

    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Define the bounds of the window
            x1 = max(x - half_window, 0)
            x2 = min(x + half_window + 1, width)
            y1 = max(y - half_window, 0)
            y2 = min(y + half_window + 1, height)

            # Extract the window
            window = image[y1:y2, x1:x2]

            # Calculate and assign the contrast value
            contrast_image[y, x] = calculate_contrast(window)

    return contrast_image


# Define window size (must be an odd number)
window_size = 5

# Compute the contrast image
contrast_image = moving_window_contrast(grayscale_image, window_size)

# Plot the contrast image
plt.figure(figsize=(10, 6))
plt.imshow(contrast_image, cmap='gray')
plt.title('Contrast Image')
plt.show()
