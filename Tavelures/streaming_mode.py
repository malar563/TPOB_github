import subprocess
import os
import pandas as pd
import re


def find_file(list_f,reg):
    for f in list_f:
        file = re.match(f,reg)
        if file is not None:
            print(file)
            return file



N_sample = 200000
client_dir = r"C:\Users\LaboPhy\Desktop\rpi\rpsa_client.exe"

data_dir = r"C:\Users\LaboPhy\Desktop\rpi\data"
reg = "bin$"

for i in range(4):
    if not os.path.exists(f"{data_dir}_{i}"):
        os.mkdir(f"{data_dir}_{i}")

    stream = subprocess.Popen([r"C:\Users\LaboPhy\Desktop\rpi\rpsa_client.exe", "-s", "-h", "10.68.9.123", "-f", "csv", "-d", f"{data_dir}_{i}", "-l", str(N_sample), "-m", "volt", "-v"])
    data = []

    while True:
        try:
            all_files = os.listdir(f"{data_dir}_{i}")
            
            file = find_file(all_files,reg)
            f = open(file)      # marche pas car les fichiers ne sont pas encore créés par Red Pitaya, donc aucun fichier match l'expression bin$ (fichier de données). Peut-être une erreur de timing où ça a pas encore été initialisé (prend des données un certain temps avant de les envoyer à l'ordi), ou Red Pitaya ne fait tout simplement qu'envoyer les données à la fin de l'acquisition complète.
            break
        except:
            pass

    while stream.poll() is None:
        data.append(f.readline())

    f.close()
    
        


        
    
