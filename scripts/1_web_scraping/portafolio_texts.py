# General libraries
import pandas as pd
import re

# Web scraping
import requests
from bs4 import BeautifulSoup

def get_content(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, features="lxml")
    content = soup.find(class_="article-content")
    text = ""
    if content:
        content = content.find_all(class_=re.compile("parrafo"))
        for paragraph in content:
            spam = paragraph.find_all("a")
            for a in spam:
                a.decompose()
            text += paragraph.get_text()
    return text

df = pd.read_csv("portafolio.csv")
df["content"] = df["link"].apply(get_content)
df.to_csv("portafolio.csv")
