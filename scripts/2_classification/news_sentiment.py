import numpy as np
import pandas as pd
import random
from transformers import pipeline
random.seed(42)

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

news.to_csv("../../data/sentiment_data.csv", index=False)

news_daily = news.groupby("date").agg(pd.Series.mode).reset_index()
news_daily.rename(columns={"score": "sentiment"}, inplace=True)

for i in range(len(news_daily)):
    curr = news_daily.loc[i, "sentiment"]
    if isinstance(curr, np.ndarray):
        news_daily.loc[i, "sentiment"] = random.choice(curr)

news_daily.to_csv("../../data/sentiment_data_daily.csv", index=False)
