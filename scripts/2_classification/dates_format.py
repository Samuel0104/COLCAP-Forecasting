import pandas as pd

months = {"Ene": "Jan",
          "Abr": "Apr",
          "Ago": "Aug",
          "Sept": "Sep",
          "Dic": "Dec"}

df1 = pd.read_csv("../../data/portafolio.csv")
df1.replace({"date": months}, regex=True, inplace=True)
df1["date"] = df1["date"].apply(lambda s: s[:5] + s[5:-8].zfill(2) + s[-8:])
df1["date"] = pd.to_datetime(df1["date"], format="%b. %d de %Y")

df2 = pd.read_csv("../../data/larepublica.csv")
df2.replace({"date": months}, regex=True, inplace=True)
df2["date"] = pd.to_datetime(df2["date"], format="%b. %d, %Y")

df = pd.concat([df1, df2], ignore_index=True).sort_values("date")
df.drop(["headline", "link"], axis=1, inplace=True)
df.to_csv("../../data/news_data.csv", index=False)
