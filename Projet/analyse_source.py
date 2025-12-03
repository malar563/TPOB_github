import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# r"Projet\no_interference.txt"
data = pd.read_csv(r"Projet\max_central\lamelle_18x18_170um(1).txt", sep = "\t", skiprows=13, decimal=",").to_numpy()
n = len(data[:,0])    # pour avoir le nb de points en fréq
c = 2.998e8     # [m/s]

lambd=data[:,0]
intensite = data[:,1]


def gaussienne(x, hauteur, position, std, offset):
    return hauteur*np.exp(-((x-position)**2)/(2*std**2))+offset

nm = np.arange(300,1000,1)
p0 = [np.max(intensite), lambd[np.argmax(intensite)], 7, 0.15]
params0, _ = curve_fit(gaussienne, lambd, intensite, p0=p0)#
hauteur0, position0, std0, offset0 = params0

print(position0, lambd[np.argmax(intensite)])


plt.plot(lambd, intensite)
plt.plot(nm, gaussienne(nm, hauteur0, position0, std0, offset0))
plt.show()


l0 = lambd[np.argmax(intensite)]    # fréquence centrale de la source
dl = data[-1,0] - data[0,0]
dw = 2*np.pi*c * (dl)/ (l0**2)
lc = 4*c*np.log(2) / dw    # difference de parcours optique doit être plus petit que ça

dt = 2*np.pi / dw
dz = c * dt     # épaisseur min qu'on peut mesurer
z = dz * n/2    # épaisseur max qu'on peut mesurer

print(lc)
print(f"dist min = {dz} nm, dist max = {z/1000} um")