# Created by alexandra at 20/12/2023
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_input(data_file):
    data = pd.read_csv(data_file, sep="\t")
    # data["intLabel"] = np.where(data["Label"]=="Cancer", 1, 0)
    data["intStudy"] = np.where(data["study"] == "NeoRheaStudy", 1, data["study"])
    data["intStudy"] = np.where(data["study"] == "PearlStudy", 2, 0)
    X1 = data["short_reads"]
    X2 = data["no_reads"]
    y = data["intStudy"]


    plt.scatter(x=X1,
                y=X2,
                c=y,
                cmap=plt.cm.RdYlBu)

    plt.show()