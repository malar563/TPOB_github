import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


petits_chemins = r"Brownien/petits_chemins.csv"

df = pd.read_csv(petits_chemins, na_values=[" "], nrows=430)


# print(np.array(df["X1"].dropna()))

for i in range(1,40):
    df1 = df[df[f"X{i}"]==df[f"X{i}"]] # Propriété des NaN : comparer un Nan avec lui même donne faux
    plt.plot(df1[f"X{i}"].values, df1[f"Y{i}"].values, ls="-", marker="None")
    # plt.scatter(df1[f"X{i}"].values, df1[f"Y{i}"].values, c=df1.index)
plt.show()

i=1
while True:
    try:
        df1 = df[df[f"X{i}"]==df[f"X{i}"]]
        x = (df1[f"X{i}"] - df1[f"X{i}"].iloc[0])
        y = (df1[f"Y{i}"] - df1[f"Y{i}"].iloc[0])
        r_2 = x**2 + y**2
        r = np.sqrt(r_2)
        print(np.mean(r))
        
    except Exception as e:
        print(e)
        break
    i+=1

