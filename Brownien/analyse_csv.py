import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


petits_chemins = r"Brownien/chemins_moins_exclusifs.csv"
# Pk ça coupe à la ligne 430?? C'est quoi ce qu'il y a après?
# Pk le tracking de particules est surtout concentré au début de la prise de données?

m_par_px = 2.18e-7
sigma_m_par_px = 0.0015e-6 # n'est pas du tout l'élément limitant ici

df = pd.read_csv(petits_chemins, na_values=[" "], nrows=300) # Parce qu'après se met à track d'autres particules (430 pour 0, )

for i in range(1,40):
    df1 = df[df[f"X{i}"]==df[f"X{i}"]] # Propriété des NaN : comparer un Nan avec lui même donne faux
    plt.plot(df1[f"X{i}"].values, df1[f"Y{i}"].values, ls="-", marker="None")
    # plt.scatter(df1[f"X{i}"].values, df1[f"Y{i}"].values, c=df1.index)
plt.show()

i=1
r2_all = []
x_all = []
y_all = []

while True:
    try:
        df1 = df[df[f"X{i}"]==df[f"X{i}"]]
        x = (df1[f"X{i}"] - df1[f"X{i}"].iloc[0])*m_par_px
        y = (df1[f"Y{i}"] - df1[f"Y{i}"].iloc[0])*m_par_px
        # print(len(y), len(x))
        r2 = x**2 + y**2

        r2_all.append(r2.reindex(df.index)) # Les nouvelles valeurs vont aller remplacer les vielles exactement au même endroit dans le df 
        x_all.append(x.reindex(df.index))
        y_all.append(y.reindex(df.index))
        
    except Exception as e:
        print(f"Erreur : {type(e).__name__} : {e}")
        break
    i+=1

# Calcul de la moyenne <r^2(t)>, <x(t)> et <y(t)>
r2_df = pd.concat(r2_all, axis=1) # r2_all est une liste de series, on la concatène avec pandas (crée un dataframe)
r2_df = r2_df[1:-1] # Enlever les NaN aux extrémités

# Enlever ceux trop court (change rien)
counts = r2_df.count(axis=0)   # nombre de valeurs non-NaN pour chaque particule
long_tracks = counts[counts > 100].index
r2_df = r2_df[long_tracks]

t = r2_df.index*0.5 # Multiplier par le temps d'exposition pour convertir en secondes
r2_mean = r2_df.mean(axis=1) # Moyenne par rapport à l'horizontale (au temps)
r2_std = r2_df.std(axis=1)
r2_count = r2_df.count(axis=1) # Nombre de particules trackées à chaque temps
r2_sem = r2_std/np.sqrt(r2_count)

x_mean, y_mean = pd.concat(x_all, axis=1).mean(axis=1), pd.concat(y_all, axis=1).mean(axis=1)
x_mean, y_mean = x_mean.iloc[1:-1], y_mean.iloc[1:-1]

plt.plot(t, r2_sem)
plt.xlabel("Temps (s)")
plt.ylabel(r"incertitude sur $\langle r^2(t) \rangle$")
plt.show()

plt.plot(t, r2_count)
plt.xlabel("Temps (s)")
plt.ylabel("Nombre de particules suivies")
plt.show()

# <x(t)> et <y(t)> selon t
plt.plot(t, y_mean.values)
plt.plot(t, x_mean.values)
plt.xlabel("Temps (s)")
plt.ylabel(r"$\langle x(t) \rangle$ ou $\langle y(t) \rangle$ ")
plt.show()


def lineaire(t, D):
    return 4*D*t

# Pour éviter les erreurs de division par 0.
mask = np.isfinite(r2_sem.values)
# mask = r2_count >= 2
mask = (r2_sem.values > 0) & np.isfinite(r2_mean.values)
t_fit = t[mask]
r2_fit = r2_mean.values[mask]
sigma_fit = r2_sem.values[mask]

params, _ = curve_fit(lineaire, t_fit, r2_fit, sigma=sigma_fit, absolute_sigma=True)
D = params[0]
sigma_D = np.sqrt(np.diag(_))[0]
print(D, sigma_D)

# <r^2(t)> selon t
plt.figure()
# plt.scatter(t, r2_mean.values, c="black")
plt.errorbar(t, r2_mean.values, yerr=r2_std.values, fmt='o', c="black", label="Données")
plt.plot(t, lineaire(t, D), c="r", label="Courbe ajustée")
plt.xlabel("Temps (s)")
plt.ylabel(r"$\langle r^2(t) \rangle$")
plt.legend()
plt.show()

def boltzmann(D, T=297, eta=0.0092, r=1e-4):
    eta = 0.92# eta = 0.01 g cm^-1 s^-1 à 293K mais diminue de 2% par K : observation à 297K
    r = 1e-6 # r = 1um à mettre en cm pour fiter avec eta
    return 6*np.pi*eta*r*D/T

print("Constante de Boltzmann :", boltzmann(D),"±", boltzmann(sigma_D))