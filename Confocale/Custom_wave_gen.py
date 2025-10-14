"""
This example shows how to program Red Pitaya to generate a custom waveform signal.
Voltage and frequency ranges depend on the Red Pitaya model.
"""
#!/usr/bin/env python3

import numpy as np
import math
from matplotlib import pyplot as plt
import redpitaya_scpi as scpi

IP = '192.168.178.102'
rp = scpi.scpi(IP)

wave_form = 'arbitrary'
freq = 10000
ampl = 1

N = 16384                   # Number of samples
t = np.linspace(0, 1, N)*2*math.pi

x = np.sin(t) + 1/3*np.sin(3*t)
y = 1/2*np.sin(t) + 1/4*np.sin(4*t)

plt.plot(t, x, t, y)
plt.title('Custom waveform')
plt.show()

rp.tx_txt('GEN:RST')

# Function for configuring a Source
rp.sour_set(1, wave_form, ampl, freq, data= x)
rp.sour_set(2, wave_form, ampl, freq, data= y)

rp.tx_txt('OUTPUT:STATE ON')
rp.tx_txt('SOUR:TRig:INT')

rp.close()