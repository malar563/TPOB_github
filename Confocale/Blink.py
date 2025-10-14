#!/usr/bin/env python3

import sys
import time
import redpitaya_scpi as scpi

IP = '10.68.9.123' #Quand vous êtes sur un ordi qui n'est pas sur le même réseau que le red pitaya, changez cette adresse IP par celle du serveur SCPI qque vous avez lancé au http://rp-f0d0d1.local/scpi_manager/ 
rp = scpi.scpi(IP)

if (len(sys.argv) > 2):
    led = int(sys.argv[2])
else:
    led = 0

print ("Blinking LED["+str(led)+"]")

period = 1 # seconds

while 1:
    time.sleep(period/2.0)
    rp.tx_txt('DIG:PIN LED' + str(led) + ',' + str(1))
    time.sleep(period/2.0)
    rp.tx_txt('DIG:PIN LED' + str(led) + ',' + str(0))

rp.close()