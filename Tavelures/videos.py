import numpy as np
import cv2
import os
from tavelures1 import calculate_contrast, moving_window_contrast


folder_path = 'THORLABS/'
types = {'lait0':'/0003.mp4','lait4':'/0004.mp4','lait8':'/0005.mp4','lait12':'/0006.mp4','lait16':'/0007.mp4'}
#types = {'papierSable':['/p60_0001.mp4','/p80_0002.mp4','/p100_0002.mp4','/p120_0001.mp4','/p600_0001.mp4','/p1200_0001.mp4']}

for t in types.keys():
    video = cv2.VideoCapture(folder_path + t + types[t])

    f = 0
    while(video.isOpened()):
        ret, frame = video.read()
        if ret is False:
            break

        path = 'THORLABS/'+f'{t}{types[t][:-4]}'
        if not os.path.exists(path):
            print('create dir')
            os.makedirs(path)
        cv2.imwrite(path + f'/00{f if f>=10 else f'0{f}'}.jpg',frame)
        f += 1
