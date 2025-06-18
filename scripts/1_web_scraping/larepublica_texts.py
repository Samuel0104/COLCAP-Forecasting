# General libraries
import pandas as pd

# Web scraping
import requests
from bs4 import BeautifulSoup

def get_content(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, features="lxml")
    
    content = soup.find(class_="html-content")
    if content:
        text = ""
        content = content.find_all("p")
        for paragraph in content:
            text += paragraph.get_text()
        return text
    return None

df = pd.read_csv("../../data/larepublica.csv")
df.dropna(inplace=True)
df["content"] = df["link"].apply(get_content)
df.to_csv("../../data/larepublica.csv", index=False)
