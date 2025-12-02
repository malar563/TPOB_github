import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("data.csv").to_numpy()
n = len(data[:,0])    # pour avoir le nb de points en fréq
c = 2.998e8     # [m/s]
l0 = ...    # fréquence centrale de la source

dw = 2*np.pi*c * (data[-1,0] - data[0,0])/l0
lc = 4*c*np.log(2) / dw    # difference de parcours optique doit être plus petit que ça

dt = 2*np.pi / dw
dz = c * dt     # épaisseur min qu'on peut mesurer
z = dz * n/2    # épaisseur max qu'on peut mesurer
