"""
This example shows how to program Red Pitaya to generate an analog 2 kHz sine wave signal with a 1 V amplitude.
Voltage and frequency ranges depend on the Red Pitaya model.
"""

# Output1 : Galvo horizontal, Output2 : Galvo vertical

import sys
import redpitaya_scpi as scpi

IP = "10.68.9.123"
rp = None

n = 512     # nombre de pixels d'un côté de l'image (ex: image de 512 x 512 pixels donne n = 512)
wave_form = 'TRIANGLE' # SINE, SQUARE, TRIANGLE, RAMPUP, RAMPDOWN, DC
freq = 175  # Hz
freq2 = freq/n  # Hz, car balaye 1024 pixels en x alors que 2 pixels en y sont balayés
ampl = 1

try:
    rp = scpi.scpi(IP)
    
    rp.tx_txt('GEN:RST')
    rp.sour_set(1, wave_form, ampl, freq2)
    rp.sour_set(2, wave_form, ampl, freq)
    rp.tx_txt('OUTPUT1:STATE ON')
    rp.tx_txt('OUTPUT2:STATE ON')
    rp.tx_txt('SOUR1:TRig:INT')
    rp.tx_txt('SOUR2:TRig:INT')
    
    print("Waveform started. Press Ctrl+C to stop")
    while True:
        pass

except KeyboardInterrupt:
    print("Exiting...")
    
finally:
    if rp is not None:
        try:
            rp.tx_txt('OUTPUT1:STATE OFF')
            rp.tx_txt('OUTPUT2:STATE OFF')
            rp.close()
            print("Connection closed")
        except Exception as e:
            print(f"Error closing connection:, {e}")

    