import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


import matplotlib as mpl
import matplotlib.font_manager as fm

def make_plots_look_good() -> None:
    """Makes the MPL rcParams paper-friendly, changing the font, the axis labels fontsize, the ticklabels fontsize, etc."""
    plt.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    plt.rc("axes", labelsize=16)

make_plots_look_good()


df = pd.read_csv('oxymetrie_data.csv', sep=",")

# Selon la saturation du vert au début du .csv, ça semble être du 19 bits : valeur de 0 à (2^19)-1

time = np.array(df["Time"])
ir = np.array(df["IR"])/2**19
red = np.array(df["Red"])/2**19
green = np.array(df["Green"])/2**19
blue = np.array(df["Blue"])/2**19

colors = [ir, red, green, blue]
plot_color = ["black", "red", "green", "blue"]
maximums = []
minimums = []
sec_par_b_medians = []

for i, color in enumerate(colors):
    maximums.append(find_peaks(color, prominence=0.001)[0])
    minimums.append(find_peaks(-color, prominence=0.001)[0])

    plt.plot(time, color, color=plot_color[i], label=(plot_color[i] if i!= 0 else "IR"))
    plt.plot(time[maximums[i]], color[maximums[i]], color=plot_color[i], marker="o", ls="None")

    time_between_peaks = time[maximums[i]][1:] - time[maximums[i]][:-1]
    sec_par_b_medians.append(np.median(time_between_peaks))

sec_par_b = np.mean(sec_par_b_medians)
i_sec_par_b = np.std(sec_par_b_medians)
bpm = 60/sec_par_b
i_bpm = bpm * i_sec_par_b / sec_par_b # Formule de Wikipedia (Propagation of uncertainties)
print("Temps d'un battement", sec_par_b, "+/-", i_sec_par_b)
print("BPM", bpm, "+/-", i_bpm)

plt.legend(loc="upper left", fontsize=14)
plt.ylabel("Intensité normalisée [-]", fontsize=14)
plt.xlabel("Temps [s]", fontsize=14)
plt.show()

# # Pandas trie la 1ere colonne en ordre croissant et les autres colonnes s'ordonnent pareil
# pd.DataFrame(np.array([temps_mins.extend(temps_maxs), intensite_mins.extend(intensite_maxs)]), columns=["temps","intensite"])
t, i = list(time[minimums[0]]), list(ir[minimums[0]])
t.extend(list(time[maximums[0]]))
i.extend(list(ir[maximums[0]]))

df_ir = pd.DataFrame(np.array([t, i]).T, columns=["temps","intensite"])
print(df_ir)

df_ir.sort_values(by="temps")
ratio_intensite = np.diff(df_ir["intensite"])



# # Comment trouver ces données? La plus haute et la plus basse de toute l'array ou bien pour un cycle?
# IminR, ImaxR = 0, 0
# IminIR, ImaxIR = 0, 0

# r=np.log(IminR/ImaxR)/np.log(IminIR/ImaxIR)


# SpO2 = ((0.81-0.18)*r)/(0.81-0.08+((0.29-0.18)*r))

# # Mettre comment calculer notre fréquence cardiaque.