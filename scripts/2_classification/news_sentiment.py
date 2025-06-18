import numpy as np
import pandas as pd
from transformers import pipeline

beto = pipeline("sentiment-analysis", model="bardsai/finance-sentiment-es-base")

news = pd.read_csv("../../data/news_data.csv")
score = []
for i in range(len(news)):
    curr = news.loc[i, "content"]
    if type(curr) is float:
        score.append(0)
    else:
        res = beto(curr, top_k=None, truncation=True)
        res = {d["label"]: d["score"] for d in res}
        top = max(res, key=res.get)
        vals = {"positive": 1,
                "negative": -1,
                "neutral": 0}
        score.append(vals[top])
news["score"] = score
news = news.drop("content", axis=1)

def get_sentiment(score):
    threshold = 0.2
    if score > threshold:
        return 1
    if score < -threshold:
        return -1
    return 0

df1 = news.groupby("date").mean().reset_index()
df1["sentiment_mean"] = df1["score"].apply(get_sentiment)

df2 = news.groupby("date").agg(pd.Series.mode).reset_index()
df1["sentiment_mode"] = df2["score"]

for i in range(len(df1)):
    curr = df1.loc[i, "sentiment_mode"]
    if isinstance(curr, np.ndarray):
        df1.loc[i, "sentiment_mode"] = df1.loc[i, "sentiment_mean"]

df1.to_csv("../../data/sentiment_data.csv", index=False)
