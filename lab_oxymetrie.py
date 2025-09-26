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

ratios_colors = []
for i, color in enumerate(colors):
    t_min = list(time[minimums[i]])
    i_min = list(color[minimums[i]])

    t_max = list(time[maximums[i]])
    i_max = list(color[maximums[i]])

    # Calcul des ratios min / max le plus proche
    ratios = []

    for t_m, i_m in zip(t_max, i_max):
        # Trouver le minimum le plus proche en temps
        idx_closest = np.argmin(np.abs(np.array(t_min) - t_m))
        i_min_closest = i_min[idx_closest]

        ratios.append(i_min_closest/i_m)
    
    ratios_colors.append(ratios)

print(ratios_colors)

def SpO2(list_ratios_IR, list_ratios_R):
    ratio_R = np.median(list_ratios_R) # IminR/ImaxR
    ratio_IR = np.median(list_ratios_IR) # IminIR/ImaxIR

    r=np.log(ratio_R)/np.log(ratio_IR)
    SpO2 = ((0.81-0.18)*r)/(0.81-0.08+((0.29-0.18)*r))

    return SpO2


print("SpO2 calculé :", SpO2(ratios_colors[0], ratios_colors[1]))



