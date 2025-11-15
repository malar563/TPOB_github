import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import astropy.units as u


petits_chemins = r"Brownien/petits_chemins_1.csv"
# Pk le tracking de particules est surtout concentré au début de la prise de données?

m_par_px = 1.724e-7 * u.m
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
        r2 = (x**2 + y**2)

        # r2_all.append(r2.reindex(df.index)) # Les nouvelles valeurs vont aller remplacer les vielles exactement au même endroit dans le df 
        # x_all.append(x.reindex(df.index))
        # y_all.append(y.reindex(df.index))

        r2_all.append(r2.reset_index(drop=True))
        x_all.append(x.reset_index(drop=True))
        y_all.append(y.reset_index(drop=True))
        
    except Exception as e:
        print(f"Erreur : {type(e).__name__} : {e}")
        break
    i+=1

# Calcul de la moyenne <r^2(t)>, <x(t)> et <y(t)>
r2_df = pd.concat(r2_all, axis=1) # r2_all est une liste de series, on la concatène avec pandas (crée un dataframe)
r2_df = r2_df[1:-1] # Enlever les NaN aux extrémités

# # Enlever ceux trop court
# counts = r2_df.count(axis=0)   # nombre de valeurs non-NaN pour chaque particule
# long_tracks = counts[counts > 100].index
# r2_df = r2_df[long_tracks]

cut_idx_above = 50
r2_df = r2_df.iloc[:cut_idx_above]

t = r2_df.index*0.5 # Multiplier par le temps d'exposition pour convertir en secondes
r2_mean = r2_df.mean(axis=1) # Moyenne par rapport à l'horizontale (au temps)
r2_std = r2_df.std(axis=1)
r2_count = r2_df.count(axis=1) # Nombre de particules trackées à chaque temps
r2_sem = r2_std/np.sqrt(r2_count)

x_mean, y_mean = pd.concat(x_all, axis=1).mean(axis=1), pd.concat(y_all, axis=1).mean(axis=1)
x_mean, y_mean = x_mean.iloc[1:-1], y_mean.iloc[1:-1]
x_mean, y_mean = x_mean.iloc[:cut_idx_above], y_mean.iloc[:cut_idx_above]

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


# def quadratique(t, a, D):
#     return a*t**2 + 4*D*t

# params, _ = curve_fit(quadratique, t_fit, r2_fit, sigma=sigma_fit, absolute_sigma=True)
# a, D = params
# print(a, D)
# # sigma_D = np.sqrt(np.diag(_))[0]
# print(D, sigma_D)



def boltzmann(D, sigma_D, T=297, eta=0.0092, r=0.5e-6):
    D = D*u.m**2/u.s
    sigma_D = sigma_D*u.m**2/u.s
    T=T*u.K
    eta = eta * u.g/(u.cm*u.s)# eta = 0.01 g cm^-1 s^-1 à 293K mais diminue de 2% par K : observation à 297K
    r = r *u.m # r = 1um à mettre en cm pour fiter avec eta

    delta_T = 2 * u.K
    delta_eta = 0.0004 * u.g/(u.cm*u.s)
    delta_r = 0.027*r

    kb = (6*np.pi*eta*r*D/T).to(u.J/u.K)   
    delta_kb = kb*np.sqrt((delta_eta/eta)**2 + (delta_r/r)**2 + (sigma_D/D)**2 + (delta_T/T)**2)

    return kb, delta_kb


kb = boltzmann(D, sigma_D)
print("Constante de Boltzmann :", kb[0],"±", kb[1])


# <r^2(t)> selon t
# Extraire mantisse et exposant
mant, exp = f"{kb[0].value:.1e}".split("e")
mant = float(mant)
exp = int(exp)
# Calculer l'incertitude sur la même échelle
unc = kb[1].value / 10**exp

# Construire la chaîne LaTeX
kb_text = fr"$k_B = ({mant:.1f} \pm {unc:.1f}) \times 10^{{{exp}}}$ J/K"

from matplotlib.ticker import ScalarFormatter
plt.figure()
plt.plot(t, r2_mean.values, c="black")
plt.fill_between(t, r2_mean.values - r2_std.values, r2_mean.values + r2_std.values, color="black", alpha=0.5, label="Incertitude (±SEM)")

# plt.errorbar(t, r2_mean.values, yerr=r2_std.values, fmt='o', c="black", label="Données")

# plt.plot(t, quadratique(t, a, D), c="r", label="Courbe ajustée")
plt.text(5, 6e-11, kb_text, horizontalalignment='center', verticalalignment='center')
plt.plot(t, lineaire(t, D), c="r", label="Courbe ajustée", linestyle="--")
plt.xlabel("Temps [s]")

ax = plt.gca() 
ax.yaxis.get_offset_text().set_visible(False)

plt.ylabel(r"$\langle r^2(t) \rangle \times 10^{11}$ [m²]")
plt.legend()
plt.show()
