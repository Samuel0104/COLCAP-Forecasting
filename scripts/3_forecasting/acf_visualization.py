import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv("../../assets/dataset.csv")

plot_acf(df["open"], lags=20, missing="drop", zero=False, auto_ylims=True, title="")
plot_acf(df["volume"], lags=20, missing="drop", zero=False, auto_ylims=True, title="")
plot_acf(df["dollar"], lags=20, missing="drop", zero=False, auto_ylims=True, title="")

plot_acf(df["open"].diff(), lags=20, missing="drop", zero=False, auto_ylims=True, title="")
plot_acf(df["volume"].diff(), lags=20, missing="drop", zero=False, auto_ylims=True, title="")
plot_acf(df["dollar"].diff(), lags=20, missing="drop", zero=False, auto_ylims=True, title="")

plt.show()