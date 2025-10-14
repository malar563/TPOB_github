import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from numpy import sqrt, pi, sum, linspace


n = 40
data = pd.read_csv(f'temps{n}.csv')

débits = data['Débit']
dyn_avg = data['Dyn_avg']
sta_avg = data['Sta_avg']
dyn_err = data['Dyn_err']
sta_err = data['Sta_err']

t = 1.013e-3
d = 5e-3
aire = pi*(d/2)**2
v = (débits[1:]/(1000*1000*60)) / aire   #divise par 1000 et 60 pour L/s = dm^3/s, donc divise par 1000 pour avoir m^3/s, total de 1000*1000*60
print(v)
def model(vdata,a,c):
    return a/sqrt(vdata*t)+c
val,cov = curve_fit(model,v,dyn_avg[1:],sigma=dyn_err[1:],absolute_sigma=True)

print("a =", val, '±', cov[0])

cap = 2
color = 'k'
f = 'o'
width = None

x = (linspace(0,16,150)/(1000*1000*60)) / aire
y = model(x,val[0],val[1])

temp = plt.errorbar(débits,dyn_avg,yerr=dyn_err,fmt=f,color=color,linestyle='none',ecolor=color,capsize=cap,elinewidth=width,label='Temporel')
modt = plt.plot(linspace(0,16,150),y,color=color,linestyle='--', label=f'Courbe ajustée : a = {round(val[0],5)} ± {round(sqrt(cov[0][0]),5)}')
#plt.errorbar(débits,sta_avg,yerr=sta_err,fmt=f,color=color,linestyle='--',ecolor=color,capsize=cap,elinewidth=width,label='Temporel (statique)')

n = ''
data = pd.read_csv(f'espace{n}.csv')
color = 'r'

débits = data['Débit']
dyn_avg = data['Dyn_avg']
sta_avg = data['Sta_avg']
dyn_err = data['Dyn_err']
sta_err = data['Sta_err']

v = (débits[1:]/(1000*1000*60)) / aire   #divise par 1000 et 60 pour L/s = dm^3/s, donc divise par 1000 pour avoir m^3/s, total de 1000*1000*60
val,cov = curve_fit(model,v,dyn_avg[1:],sigma=dyn_err[1:],absolute_sigma=True)

print("a =", val[0], '±', cov[0][0])

x = (linspace(0,16,150)/(1000*1000*60)) / aire
y = model(x,val[0],val[1])

spac = plt.errorbar(débits,dyn_avg,yerr=dyn_err,fmt=f,color=color,linestyle='none',ecolor=color,capsize=cap,elinewidth=width,label='Spatial')
mods = plt.plot(linspace(0,16,150),y,color=color,linestyle='--', label=f'Courbe ajustée : a = {round(val[0],5)} ± {round(sqrt(cov[0][0]),5)}')
#plt.errorbar(débits,sta_avg,yerr=sta_err,fmt=f,color=color,linestyle='--',ecolor=color,capsize=cap,elinewidth=width,label='Spatial (statique)')
plt.xlabel('Débit [mL/min]')
plt.ylabel('Contraste [-]')
plt.legend()
plt.show()