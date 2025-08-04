import pandas as pd

prices = pd.read_excel("../../data/prices_data.xlsx").dropna() # Erase 2020-03-13
prices.sort_values("date", inplace=True, ignore_index=True)

dollar = pd.read_excel("../../data/trm.xlsx")
dollar.rename(columns={"rate": "dollar"}, inplace=True)

sents = pd.read_csv("../../data/sentiment_data_daily.csv")
sents["date"] = pd.to_datetime(sents["date"], format="%Y-%m-%d")

df = prices.merge(dollar, how="left", on="date")
df = df.merge(sents, how="right", on="date")
df.to_csv("../../assets/dataset.csv", index=False)
