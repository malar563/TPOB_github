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


# thickness(r"Projet\max_central\max_central_OPDnul.txt", r"Projet\max_central\lamelle_22x22_130um(1).txt")
# thickness(r"Projet\off-set_avance\off-set_avance.txt", r"Projet\off-set_avance\lamelle_22x22_130um_1.txt")


# ===============================================================================
# TENTATIVE MARYLISE

from scipy.signal.windows import boxcar
from pylab import r_
def smooth(x, smoothing_param=3):
    window_len=smoothing_param*2+1
    s=r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w=boxcar(smoothing_param*2+1)
    y=np.convolve(w/np.sum(w),s,mode='valid')
    return y[smoothing_param:-smoothing_param] 


def get_opd(file_0 : str, file_data : str):
    data_0 = pd.read_csv(file_0, sep = "\t", skiprows=13, decimal=",").to_numpy()      # pas d'échantillon
    data_1 = pd.read_csv(file_data, sep = "\t", skiprows=13, decimal=",").to_numpy()  # échantillon

    wavelenghts = data_0[:,0]*1e-9
    ref = data_0[:,1]
    sample = data_1[:,1]
    c = 299792458
    # plt.plot(wavelenghts, ref, label="no sample data")
    # plt.plot(wavelenghts, sample, label="sample data")
    # plt.xlabel("wavelengths [m]")
    # plt.ylabel("Intensity [-]")
    # plt.legend()
    # plt.show()

    freqs = 2*np.pi*c / wavelenghts
    w_uniform = np.linspace(freqs.min(), freqs.max(), len(freqs))
    I_w_ref = np.interp(w_uniform, np.sort(freqs), ref)
    I_w_sample = np.interp(w_uniform, np.sort(freqs), sample)

    # Ordonner omega en ordre croissant avec les intensités associées 
    idx = np.argsort(freqs)
    omega_sorted = freqs[idx]
    ref_sorted = ref[idx]
    sample_sorted = sample[idx]

    # pas uniforme pour omega (apparemment nécessaire pour fft?)
    M = len(omega_sorted)
    omega_uniform = np.linspace(omega_sorted.min(), omega_sorted.max(), M)

    from scipy.interpolate import interp1d
    # Interpoler omega selon le pas uniforme
    interp_ref = interp1d(omega_sorted, ref_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    interp_sample = interp1d(omega_sorted, sample_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    I_w_ref = interp_ref(omega_uniform)
    I_w_sample = interp_sample(omega_uniform)

    # Enlever l'enveloppe dûe à la source
    # plt.plot(omega_sorted, ref_sorted, label="no sample data")
    # plt.plot(omega_sorted, sample_sorted, label="sample data")
    # plt.plot(omega_uniform, I_w_ref, label="no sample interpolation")
    # plt.plot(omega_uniform, I_w_sample, label="sample interpolation")
    # plt.xlabel("fréquences omega [s^-1]")
    # plt.ylabel("Intensity [-]")
    # plt.legend()
    # plt.show()

    envelope = smooth(I_w_ref, smoothing_param=12)
    I_mod_ref = I_w_ref - envelope
    # plt.plot(w_uniform, envelope, label="no sample interpolation smoothing (enveloppe)")
    # plt.plot(w_uniform, I_w_ref, label="no sample interpolation")
    # plt.plot(w_uniform, I_mod_ref, label="difference between them")
    # plt.xlabel("fréquences omega [s^-1]")
    # plt.ylabel("Intensity [-]")
    # plt.legend()    
    # plt.show()

    envelope =  smooth(I_w_sample, smoothing_param=12)
    I_mod_sample = I_w_sample - envelope
    # plt.plot(w_uniform, envelope, label="sample interpolation smoothing (enveloppe)")
    # plt.plot(w_uniform, I_w_sample, label="sample interpolation")
    # plt.plot(w_uniform, I_mod_sample, label="difference between them")
    # plt.xlabel("fréquences omega [s^-1]")
    # plt.ylabel("Intensity [-]")
    # plt.legend()    
    # plt.show()

    # fft, calcul des OPD et d
    opt_ref = np.fft.fft(I_mod_ref)
    opt_sample = np.fft.fft(I_mod_sample)
    n = len(opt_ref)
    domega = w_uniform[1] - w_uniform[0]
    tau = np.fft.fftfreq(n, domega)
    OPD = c*tau
    positif = OPD>0
    OPD = OPD[positif]
    opt_ref = np.abs(opt_ref[positif])
    opt_sample = np.abs(opt_sample[positif])

    xn=OPD[np.argmax(opt_ref)]
    xa=OPD[np.argmax(opt_sample)]
    #d=(xa-xn)/(1.4995-1)
    print(xa, xn)

    plt.plot(OPD*1e6, opt_ref, label="No sample")
    plt.plot(OPD*1e6, opt_sample, label="Sample")
    plt.legend()
    plt.xlabel("Optical path [um]")
    plt.ylabel("Intensity [-]")
    plt.show()

    return (xn, xa)


# thickness(r"Projet\max_central\max_central_OPDnul.txt", r"Projet\max_central\lamelle_22x22_130um(1).txt")
# thickness(r"Projet\off-set_avance\off-set_avance.txt", r"Projet\off-set_avance\lamelle_22x22_130um_1.txt")

xnc, xac = get_opd(r"Projet\max_central\max_central_OPDnul.txt", r"Projet\max_central\lamelle_18x18_170um(1).txt")
xno, xao = get_opd(r"Projet\off-set_avance\off-set_avance.txt", r"Projet\off-set_avance\lamelle_18x18_170um_1.txt")
d = 170e-6
ng_c = (xac-xnc)/d + 1
ng_o = (xao-xno)/d + 1
ng = np.mean([ng_c, ng_o])
print(ng_c, ng_o, ng)

xnc, xac = get_opd(r"Projet\max_central\max_central_OPDnul.txt", r"Projet\max_central\lamelle_22x22_130um(1).txt")
xno, xao = get_opd(r"Projet\off-set_avance\off-set_avance.txt", r"Projet\off-set_avance\lamelle_22x22_130um_1.txt")
d_n=(xac-xnc)/(ng-1)
d_o=(xao-xno)/(ng-1)
d = np.mean([d_n, d_o])
print(d_n*1e6, d_o*1e6, d*1e6)

















