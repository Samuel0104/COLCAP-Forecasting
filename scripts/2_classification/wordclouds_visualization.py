import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from unidecode import unidecode
from wordcloud import WordCloud

spacy.cli.download("es_core_news_sm")
nlp = spacy.load("es_core_news_sm",
                 exclude=["tok2vec", "morphologizer", "parser", "senter",
                          "attribute_ruler", "lemmatizer", "ner"])
pat = re.compile(r"[^a-z ]")
spaces = re.compile(r"\s{2,}")

def preprocess(text, nlp):
    norm_text = unidecode(text).lower()
    clean_text = re.sub(pat, " ", norm_text)
    spaces_text = re.sub(spaces, " ", clean_text)
    tokens = nlp(spaces_text)
    filtered_tokens = filter(lambda token: len(token) > 3 and not token.is_stop, tokens)
    filtered_text = " ".join(token.text for token in filtered_tokens)
    return filtered_text.strip()

def bag_of_words(preprocessed_corpus):
    vect = TfidfVectorizer(max_features=2000, max_df=0.4, sublinear_tf=True).fit(preprocessed_corpus)
    X = vect.transform(preprocessed_corpus).toarray()
    return X, vect

def get_wordcloud(X, vect):
    counts = X.sum(axis=0)
    vocab = vect.get_feature_names_out()
    counts_dict = {word: count for (word, count) in zip(vocab, counts)}
    wc = WordCloud(width=500, height=400, background_color="white").generate_from_frequencies(counts_dict)
    return wc

def get_wordcloud_sentiment(X, vect, sents, cat):
    mask = np.array(list(map(lambda y: y == cat, sents)))
    X = X[mask, :]
    wc = get_wordcloud(X, vect)
    return wc

news = pd.read_csv("../../data/news_data.csv")
news.dropna(inplace=True)

preprocessed_text = news["content"].apply(preprocess, nlp=nlp)
X, vect = bag_of_words(preprocessed_text)
sents = news["score"].to_list()

# Negative news
fig, ax = plt.subplots()
wc = get_wordcloud_sentiment(X, vect, sents, -1)
ax.imshow(wc)
ax.axis("off")

# Positive news
fig, ax = plt.subplots()
wc = get_wordcloud_sentiment(X, vect, sents, 1)
ax.imshow(wc)
ax.axis("off")