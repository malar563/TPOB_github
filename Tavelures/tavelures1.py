import matplotlib.pyplot as plt
import numpy as np
import cv2


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


if __name__=='__main__':
    #types = {'alu':range(1,10),'gant':range(32,41),'leaf1':range(11,20),'leaf2':range(22,31),'metalnoir':range(19,28),'mousse':range(51,60),'nemesis':range(41,50),'papier':range(29,38),'sable':range(61,70)}
    #contrastes = {'alu':(), 'gant':(), "leaf1":(),'leaf2':(),'metalnoir':(),'mousse':(),'nemesis':(),'papier':(),'sable':()}
    #types = {'lait1/0ml/0011':range(100,301),'lait1/9ml/0010':range(100,301),'lait1/16ml/0005':range(100,301)}
    #contrastes = {'lait1/0ml/0011':(),'lait1/9ml/0010':(),'lait1/16ml/0005':()}
    types = [60,80,100,120,600,1200]
    contrastes = {60:(),80:(),100:(),120:(),600:(),1200:()}
    # Load the image
    folder_path = 'THORLABS/papierSable/'

    for t in types:
        type_path = folder_path + f'p{t}/'
        type_contrast = []
        for i in range(20):
            image_path = type_path + f'00{f'0{i}' if i<10 else i}.jpg'
            image = cv2.imread(image_path)


            # Convert the image to grayscale
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            #display the gray scale image
            #plt.figure()
            #plt.imshow(grayscale_image,cmap = 'gray')
            #plt.title('grayscale image')
            #plt.show()

            # calculating contrast of the image

            average = np.mean(grayscale_image)
            standardDeviation = np.std(grayscale_image)
            contrast = standardDeviation / average
            print(f'contrast {i} :', contrast)
            type_contrast.append(contrast)

            # Compute the histogram
            #histogram = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])

            # Plot the histogram
            #plt.figure(figsize=(10, 6))
            #plt.plot(histogram, color='black')
            #plt.title('Grayscale Histogram')
            #plt.xlabel('Pixel Intensity')
            #plt.ylabel('Frequency')
            #plt.grid(True)
            #plt.show()

            # Define window size (must be an odd number)
            #window_size = 5

            # Compute the contrast image
            #contrast_image = moving_window_contrast(grayscale_image, window_size)

            # Plot the contrast image
            #plt.figure(figsize=(10, 6))
            #plt.imshow(contrast_image, cmap='gray')
            #plt.title('Contrast Image')
            #plt.show()
        
        contrastes[t] = (float(round(np.mean(type_contrast),3)),float(round(np.std(type_contrast),3)))

    print(contrastes)
