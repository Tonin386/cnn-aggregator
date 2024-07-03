from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import plotly.graph_objects as go
from textblob import TextBlob
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import string
import nltk
import re

from .models import Article

nltk.download('punkt')
nltk.download('stopwords')

def retrieve_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    return soup

def read_article(article_html):
    title = article_html.find('h1').get_text().replace("\n", "").replace("  ", "")
    texts = article_html.find_all('p', class_='paragraph')
    texts = "".join([text.get_text().replace("  ", "") for text in texts])
    return title, texts

def save_article(title, topic, content, src, pol_score, sbj_score):
    article_dict = [title, topic, content, src, pol_score, sbj_score]
    Article.objects.create(
        title=title,
        topic=topic,
        content=content,
        source=src,
        polarity=pol_score,
        subjectivity=sbj_score
    )

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def analyze_sentiment(text):
    blob = TextBlob(text)

    disturbing = [
        "apocalypse", "disaster", "crisis", "catastrophe", "threat",
        "terror", "violence", "fear", "destruction", "chaos", "danger",
        "terrible", "atrocious", "calamity", "tragedy", "carnage", 'murder',
        "war", "menace", "critical"
    ]

    negative = [
        "despair", "suffering", "misery", "anguish", "pain",
        "grief", "sadness", "melancholy", "depression", "anxiety",
        "issue", "problem", "bad", "vulnerable", "sad", "concern",
        "concerning", "worried"
    ]

    neutral = [
        "routine", "commonplace", "ordinary", "standard", "typical",
        "usual", "unremarkable", "mundane", "average", "common", "boring"
    ]

    positive = [
        "hope", "comfort", "relief", "contentment", "satisfaction",
        "joy", "happiness", "love", "excitement", "enthusiasm", "good",
        "agreement", "effort", "happy", "hopeful", "proud"
    ]

    optimistic = [
        "potential", "prosperity", "success", "wonderful", "perfect", 
        "achievement", "growth", "fulfillment", "bliss", "euphoria", "peace",
        "extasis", "successful", "greatness", "incredible"
    ]

    custom_score = 0

    for word in text.split(" "):
        if word in string.punctuation + "”“’–":
            continue
        word = word.lower()
        if word in disturbing:
            custom_score -= 0.1
        elif word in negative:
            custom_score -= 0.05
        elif word in neutral:
            custom_score -= custom_score / 10
        elif word in positive:
            custom_score += 0.05
        elif word in optimistic:
            custom_score += 0.1
    
    sentiment_polarity = np.tanh(10 * (0.25 * blob.sentiment.polarity + custom_score))
    sentiment_subjectivity = blob.sentiment.subjectivity
    
    return sentiment_polarity, sentiment_subjectivity

def retrieve_cnn_homepage():
    url = 'https://edition.cnn.com/'

    soup = retrieve_webpage(url)

    link_date_regex = r"20[0-9][0-9](\/[0-9][0-9]){2}\/"
    headline = soup.find('a', class_="container_lead-package__title-url")
    link = headline.get_attribute_list("href")[0][1:]
    topic = re.sub(link_date_regex, "", link).split("/")[0]
    full_link = f'{url}{link}'

    articles_match = Article.objects.filter(source=full_link)

    if len(articles_match) == 0:
        title, content = read_article(retrieve_webpage(full_link))
        polarity_score, subjectivity_score = analyze_sentiment(preprocess_text(content))
        save_article(title, topic, content, full_link, polarity_score, subjectivity_score)
    
    articles = soup.find_all('a', class_='container__link--type-article')
    print("Articles found:", len(articles))

    for article in articles:
        link = article.get_attribute_list("href")[0][1:]
        topic = re.sub(link_date_regex, "", link).split("/")[0]
        full_link = f'{url}{link}'

        articles_match = Article.objects.filter(source=full_link)
        if len(articles_match) == 0:
            title, content = read_article(retrieve_webpage(full_link))
            polarity_score, subjectivity_score = analyze_sentiment(preprocess_text(content))
            save_article(title, topic, content, full_link, polarity_score, subjectivity_score)
