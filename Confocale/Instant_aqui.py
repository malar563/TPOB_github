"""
This example shows how to instantly acquire 16k samples of a signal on fast analog inputs. 
The time length of the acquired signal depends on the time scale of a buffer that can be set with a decimation factor. 
The decimations and time scales of a buffer are given in the sample rate and decimation. 
Voltage and frequency ranges depend on the Red Pitaya model.
"""
#!/usr/bin/env python3

import sys
import redpitaya_scpi as scpi
import matplotlib.pyplot as plot

IP = 'rp-f066c8.local'

rp = scpi.scpi(IP)

rp.tx_txt('ACQ:RST')

rp.tx_txt('ACQ:DEC 4')
rp.tx_txt('ACQ:START')
rp.tx_txt('ACQ:TRig NOW')

while 1:
    rp.tx_txt('ACQ:TRig:STAT?')
    if rp.rx_txt() == 'TD':
        break

## ! OS 2.00 or higher only ! ##
while 1:
    rp.tx_txt('ACQ:TRig:FILL?')
    if rp.rx_txt() == '1':
        break

rp.tx_txt('ACQ:SOUR1:DATA?')
buff_string = rp.rx_txt()
buff_string = buff_string.strip('{}\n\r').replace("  ", "").split(',')
buff = list(map(float, buff_string))

plot.plot(buff)
plot.ylabel('Voltage')
plot.show()