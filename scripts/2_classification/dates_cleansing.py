from datetime import timedelta
import pandas as pd

def replace_weekends(date):
    weekday = date.weekday()
    if weekday == 5:
        return date + timedelta(days=-1)
    if weekday == 6:
        return date + timedelta(days=-2)
    return date

def replace_holiday(date):
    weekday = date.weekday()
    if weekday == 0:
        return date + timedelta(days=-3)
    return date + timedelta(days=-1)

news = pd.read_csv("../../data/news_data.csv")
news.replace({"date": {"2025": "2024"}}, regex=True, inplace=True) # Replace 2025 for 2024
news["date"] = pd.to_datetime(news["date"], format="%Y-%m-%d")
news["date"] = news["date"].apply(replace_weekends) # Replace weekends

prices = pd.read_excel("../../data/prices_data.xlsx")
prices.sort_values("date", inplace=True, ignore_index=True)

# Replace holidays
trading_days = prices["date"].values
for i in range(len(news)):
    while news.loc[i, "date"] not in trading_days:
        news.loc[i, "date"] = replace_holiday(news.loc[i, "date"])

# Fill missing dates
start = min(news["date"])
end = max(news["date"])
for i in range(len(prices)):
    curr = prices.loc[i, "date"]
    if curr > start and curr < end:
        if curr not in news["date"].values:
            news.loc[len(news)] = [curr, None]

news.sort_values("date", inplace=True, ignore_index=True)
news.to_csv("../../data/news_data.csv", index=False)
