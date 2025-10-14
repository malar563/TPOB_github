import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd


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
    #types = [60,80,100,120,600,1200]
    #contrastes = {60:(),80:(),100:(),120:(),600:(),1200:()}
    types = {0:'lait0/0003',4:'lait4/0004',8:'lait8/0005',12:'lait12/0006',16:'lait16/0007'}
    intensité = np.zeros((2,5,500))
    temps = np.linspace(0,14.5,500)
    n = 40
    contrast = np.zeros((2,5,n))
    dyn_avg = []
    sta_avg = []
    dyn_err = []
    sta_err = []
    # Load the image
    folder_path = 'THORLABS/'

    for m in range(n):
        di,dj = np.random.randint(739,772),np.random.randint(385,445)
        si,sj = np.random.randint(605,635),np.random.randint(390,450)

        for t in types:
            type_path = folder_path + f'{types[t]}/'
            idx = round(t/4)

            for i in range(500):
                image_path = type_path + f'00{f'0{i}' if i<10 else i}.jpg'
                image = cv2.imread(image_path)
                dynamic = image[dj,di]
                static = image[sj,si]
                intensité[0][idx][i] = dynamic[0]
                intensité[1][idx][i] = static[0]

            average = np.mean(intensité[0][idx])
            standardDeviation = np.std(intensité[0][idx])
            contrast[0][idx][m] = standardDeviation / average
            average = np.mean(intensité[1][idx])
            standardDeviation = np.std(intensité[1][idx])
            contrast[1][idx][m] = standardDeviation / average
        
    for t in types:
        idx = round(t/4)
        dyn_avg.append(np.mean(contrast[0][idx]))
        sta_avg.append(np.mean(contrast[1][idx]))
        dyn_err.append(np.std(contrast[0][idx]))
        sta_err.append(np.std(contrast[1][idx]))


    df = pd.DataFrame({'Débit':np.linspace(0,16,5),'Dyn_avg':dyn_avg,'Sta_avg':sta_avg,'Dyn_err':dyn_err,'Sta_err':sta_err})
    df.to_csv(f'temps{n}.csv',index=False)
            
    plt.errorbar(np.linspace(0,16,5),dyn_avg,yerr=dyn_err,capsize=2)
    plt.errorbar(np.linspace(0,16,5),sta_avg,yerr=sta_err,capsize=2)
    plt.xlabel('Débit de la solution [ml/min]')
    plt.ylabel("Contraste temporel [-]")
    plt.show()