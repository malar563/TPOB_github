import pandas as pd
import numpy as np


df = pd.read_csv('jspkoi.csv')

array_1 = np.array([])/1023
array_1 = np.array([])/1023

# Comment trouver ces données? La plus haute et la plus basse de toute l'array ou bien pour un cycle?
IminR, ImaxR = 0, 0
IminIR, ImaxIR = 0, 0

r=np.log(IminR/ImaxR)/np.log(IminIR/ImaxIR)

SpO2 = ((0.81-0.18)*r)/(0.81-0.08+((0.29-0.18)*r))

# Mettre comment calculer notre fréquence cardiaque.