import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def thickness(file_0 : str, file_data : str):
    data_0 = pd.read_csv(file_0, sep = "\t", skiprows=13, decimal=",").to_numpy()      # pas d'échantillon
    data_1 = pd.read_csv(file_data, sep = "\t", skiprows=13, decimal=",").to_numpy()  # échantillon

    wavelenghts = data_0[:,0]
    ref = data_0[:,1]
    sample = data_1[:,1]
    c = 299792458

    freqs = 2*np.pi*c / wavelenghts

    opt_ref = np.fft.fft(ref)
    opt_sample = np.fft.fft(sample)
    n = len(opt_ref)
    opt_path = np.fft.fftfreq(n, abs(freqs[1] - freqs[0]))
    opt_path = np.abs(opt_path[:n//2])
    opt_ref = opt_ref[:n//2]
    opt_sample = opt_sample[:n//2]

    plt.plot(opt_path*1e6, np.abs(opt_ref), label="No sample")
    plt.plot(opt_path*1e6, np.abs(opt_sample), label="Sample")
    plt.legend()
    plt.xlabel("Optical path [um]")
    plt.ylabel("Intensity [-]")
    plt.show()




thickness(r"Projet\max_central\max_central_OPDnul.txt", r"Projet\max_central\lamelle_22x22_130um(1).txt")
thickness(r"Projet\off-set_avance\off-set_avance.txt", r"Projet\off-set_avance\lamelle_22x22_130um_1.txt")