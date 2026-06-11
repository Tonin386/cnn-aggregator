from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import string
import nltk
import re
import os
import json
import html
import unicodedata
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

from .models import Article

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs
    DetectorFactory.seed = 0
except ImportError:
    LangDetectException = Exception
    detect_langs = None

def nltk_resource_available(path):
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        if os.environ.get("CNN_AGGREGATOR_DOWNLOAD_NLTK") == "1":
            nltk.download(path.split("/")[-1], quiet=True)
            try:
                nltk.data.find(path)
                return True
            except LookupError:
                return False
        return False

PUNKT_AVAILABLE = nltk_resource_available("tokenizers/punkt")
STOPWORDS_AVAILABLE = nltk_resource_available("corpora/stopwords")
VADER_AVAILABLE = nltk_resource_available("sentiment/vader_lexicon.zip")

try:
    VADER_ANALYZER = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
except LookupError:
    VADER_ANALYZER = None

LEXICON_DIR = Path(__file__).resolve().parent / "lexicons"

def load_json_lexicon(filename, fallback):
    try:
        with (LEXICON_DIR / filename).open(encoding="utf-8") as lexicon_file:
            return json.load(lexicon_file)
    except (OSError, json.JSONDecodeError):
        return fallback

def build_sentiment_lexicon(categories):
    lexicon = {}
    for category in categories.values():
        score = np.clip(float(category["score"]) * 1.18, -1, 1)
        for word in category["words"]:
            lexicon[word.lower()] = score
    return lexicon

def build_phrase_lexicon(entries):
    return {
        entry["pattern"]: np.clip(float(entry["score"]) * 1.18, -1, 1)
        for entry in entries
    }

SENTIMENT_WORD_CATEGORIES = load_json_lexicon("sentiment_words.json", {})
SENTIMENT_LEXICON = build_sentiment_lexicon(SENTIMENT_WORD_CATEGORIES)
NEWS_PHRASE_LEXICON = build_phrase_lexicon(load_json_lexicon("news_phrases.json", []))
MODIFIERS = load_json_lexicon("modifiers.json", {})
SUBJECTIVITY_CUES = load_json_lexicon("subjectivity_cues.json", {})

NEGATIONS = set(MODIFIERS.get("negations", []))
INTENSIFIERS = {word: float(value) for word, value in MODIFIERS.get("intensifiers", {}).items()}
DIMINISHERS = {word: float(value) for word, value in MODIFIERS.get("diminishers", {}).items()}
OBJECTIVE_CUES = set(SUBJECTIVITY_CUES.get("objective_cues", []))
SUBJECTIVE_CUES = set(SUBJECTIVITY_CUES.get("subjective_cues", []))
OBJECTIVE_PHRASES = SUBJECTIVITY_CUES.get("objective_phrases", [])
SUBJECTIVE_PHRASES = SUBJECTIVITY_CUES.get("subjective_phrases", [])
SUBJECTIVITY_OBJECTIVE_THRESHOLD = 0.30
SUBJECTIVITY_SUBJECTIVE_THRESHOLD = 0.50
OBJECTIVE_PHRASE_SIGNAL_WEIGHT = 1.35
SUBJECTIVE_PHRASE_SIGNAL_WEIGHT = 1.45
EVIDENCE_SIGNAL_WEIGHT = 0.45
SENTIMENT_HIT_SIGNAL_WEIGHT = 0.40
POLARITY_SIGNAL_WEIGHT = 0.70
OBJECTIVE_SUBJECTIVITY_WEIGHT = 0.24
SUBJECTIVE_SUBJECTIVITY_WEIGHT = 0.36
QUOTE_OR_NUMBER_PATTERN = re.compile(r'["“”]|\b\d+(?:\.\d+)?%?\b')
WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?%?")
ENGLISH_STOPWORD_FALLBACK = {
    "the", "and", "that", "have", "for", "not", "with", "you", "this", "but",
    "his", "from", "they", "say", "her", "she", "will", "one", "all", "would",
    "there", "their", "what", "about", "which", "when", "make", "can", "said",
    "who", "more", "if", "out", "up", "into", "than", "them", "its", "also",
}
CNN_BASE_URL = "https://www.cnn.com/"
CNN_ARTICLE_SITEMAP_INDEX_URL = "https://www.cnn.com/sitemap/article.xml"
AL_JAZEERA_ARTICLE_ARCHIVE_INDEX_URL = "https://www.aljazeera.com/sitemaps/article-archive.xml"
AL_JAZEERA_ARTICLE_NEW_INDEX_URL = "https://www.aljazeera.com/sitemaps/article-new.xml"
BBC_ARCHIVE_INDEX_URL = "https://www.bbc.com/sitemaps/https-index-com-archive.xml"
BBC_NEWS_INDEX_URL = "https://www.bbc.com/sitemaps/https-index-com-news.xml"
NPR_STANDARD_INDEX_URL = "https://googlecrawl.npr.org/standard/sitemap_index.xml"
NPR_NEWS_SITEMAP_URL = "https://googlecrawl.npr.org/news/sitemap_news.xml"
GUARDIAN_SEARCH_API_URL = "https://content.guardianapis.com/search"
GUARDIAN_API_KEY = os.environ.get("GUARDIAN_API_KEY", "test")
HISTORICAL_NEWS_SOURCES = [
    {"name": "CNN", "kind": "cnn"},
    {"name": "Al Jazeera", "kind": "al_jazeera"},
    {"name": "NPR", "kind": "npr"},
    {"name": "The Guardian", "kind": "guardian"},
]
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
}
ARTICLE_DATE_PATTERN = re.compile(r"/(?P<year>20\d{2})/(?P<month>\d{2})/(?P<day>\d{2})/")
LOOSE_ARTICLE_DATE_PATTERN = re.compile(r"/(?P<year>20\d{2})/(?P<month>\d{1,2})/(?P<day>\d{1,2})(?:/|$)")
NPR_STANDARD_SITEMAP_PATTERN = re.compile(r"sitemap_standard_01-(?P<half>Jan|Jul)-(?P<year>\d{2})\.xml$")
LINK_DATE_REGEX = r"20[0-9][0-9](\/[0-9][0-9]){2}\/"

def article_highlight_word_groups():
    groups = {
        "disturbing": set(),
        "negative": set(),
        "neutral": set(),
        "positive": set(),
        "optimistic": set(),
    }

    for word, score in SENTIMENT_LEXICON.items():
        if score <= -0.75:
            groups["disturbing"].add(word)
        elif score < -0.05:
            groups["negative"].add(word)
        elif score <= 0.05:
            groups["neutral"].add(word)
        elif score < 0.65:
            groups["positive"].add(word)
        else:
            groups["optimistic"].add(word)

    return groups

def retrieve_webpage(url):
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()
    html_text = response.content.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html_text, 'html.parser')

    return soup

MOJIBAKE_MARKERS = (
    "‚", "Ä", "Ã", "Â", "â€", "â€™", "â€œ", "â€", "â€“", "â€”",
)
MOJIBAKE_REPLACEMENTS = {
    "‚Äô": "’",
    "‚Äò": "‘",
    "‚Äú": "“",
    "‚Äù": "”",
    "‚Äî": "—",
    "‚Äì": "–",
    "‚Ä¶": "…",
    "‚Ä¢": "•",
    "‚Ç¨": "€",
    "â€™": "’",
    "â€˜": "‘",
    "â€œ": "“",
    "â€": "”",
    "â€“": "–",
    "â€”": "—",
    "â€¦": "…",
    "â€¢": "•",
    "Â ": " ",
    "Â": "",
    "Ã©": "é",
    "Ã¨": "è",
    "Ãª": "ê",
    "Ã«": "ë",
    "Ã¡": "á",
    "Ã ": "à",
    "Ã¢": "â",
    "Ã¤": "ä",
    "Ã­": "í",
    "Ã®": "î",
    "Ã³": "ó",
    "Ã´": "ô",
    "Ã¶": "ö",
    "Ãº": "ú",
    "Ã¼": "ü",
    "Ã±": "ñ",
    "Ã§": "ç",
}

def fix_text_encoding(text):
    if not text:
        return text

    cleaned_text = html.unescape(str(text)).replace("\xa0", " ")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    for broken_text, fixed_text in MOJIBAKE_REPLACEMENTS.items():
        cleaned_text = cleaned_text.replace(broken_text, fixed_text)

    if any(marker in cleaned_text for marker in MOJIBAKE_MARKERS):
        for source_encoding in ("macroman", "latin1", "cp1252"):
            try:
                repaired_text = cleaned_text.encode(source_encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if mojibake_score(repaired_text) < mojibake_score(cleaned_text):
                cleaned_text = repaired_text
                break

    return unicodedata.normalize("NFC", cleaned_text)

def mojibake_score(text):
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)

def repeated_term_weight(term_counts, term):
    term_counts[term] = term_counts.get(term, 0) + 1
    return 1 / term_counts[term]

def clean_phrase_pattern(pattern):
    return pattern.replace("\\b", "").replace("\\", "").strip()

def add_sentiment_contribution(contributions, term, kind, score, weight, modifiers=None):
    entry = contributions.setdefault(
        f"{kind}:{term}",
        {
            "term": term,
            "kind": kind,
            "occurrences": 0,
            "total_weight": 0.0,
            "base_score_total": 0.0,
            "contribution": 0.0,
            "modifiers": set(),
        },
    )
    entry["occurrences"] += 1
    entry["total_weight"] += weight
    entry["base_score_total"] += score
    entry["contribution"] += score * weight
    if modifiers:
        entry["modifiers"].update(modifiers)

def add_subjectivity_contribution(contributions, term, kind, score, weight, modifiers=None, count_occurrence=True):
    entry = contributions.setdefault(
        f"{kind}:{term}",
        {
            "term": term,
            "kind": kind,
            "occurrences": 0,
            "total_weight": 0.0,
            "base_score_total": 0.0,
            "contribution": 0.0,
            "subjectivity_contribution": 0.0,
            "modifiers": set(),
        },
    )
    entry.setdefault("subjectivity_contribution", 0.0)
    if count_occurrence:
        entry["occurrences"] += 1
        entry["total_weight"] += weight
    entry["subjectivity_contribution"] += score * weight
    if modifiers:
        entry["modifiers"].update(modifiers)

def weighted_token_count(tokens, cue_words, prefix):
    term_counts = {}
    total = 0.0
    for token in tokens:
        if token in cue_words:
            total += repeated_term_weight(term_counts, f"{prefix}:{token}")
    return total

def weighted_phrase_count(normalized_text_lower, phrase_patterns, prefix):
    term_counts = {}
    total = 0.0
    for pattern in phrase_patterns:
        occurrences = len(re.findall(pattern, normalized_text_lower))
        for _ in range(occurrences):
            total += repeated_term_weight(term_counts, f"{prefix}:{pattern}")
    return total

def subjectivity_lexicon_balance():
    objective_mass = OBJECTIVE_SUBJECTIVITY_WEIGHT * (
        len(OBJECTIVE_CUES) + OBJECTIVE_PHRASE_SIGNAL_WEIGHT * len(OBJECTIVE_PHRASES)
    )
    subjective_mass = SUBJECTIVE_SUBJECTIVITY_WEIGHT * (
        len(SUBJECTIVE_CUES) + SUBJECTIVE_PHRASE_SIGNAL_WEIGHT * len(SUBJECTIVE_PHRASES)
    )

    if objective_mass == 0:
        return 1.0
    return subjective_mass / objective_mass

SUBJECTIVITY_OBJECTIVE_BALANCE = subjectivity_lexicon_balance()

def classify_subjectivity(score):
    score = np.clip(float(score), 0.0, 1.0)
    if score < SUBJECTIVITY_OBJECTIVE_THRESHOLD:
        return "Objectif"
    if score < SUBJECTIVITY_SUBJECTIVE_THRESHOLD:
        return "Insuffisamment objectif"
    return "Subjectif"

def language_probabilities(text):
    if not text or not text.strip() or detect_langs is None:
        return []

    try:
        return detect_langs(text[:5000])
    except LangDetectException:
        return []

def is_english_text(text, min_probability=0.78):
    normalized_text = re.sub(r"\s+", " ", text or "").strip()
    tokens = WORD_PATTERN.findall(normalized_text.lower())
    alphabetic_tokens = [token for token in tokens if any(character.isalpha() for character in token)]

    if len(alphabetic_tokens) < 12:
        return True

    probabilities = language_probabilities(normalized_text)
    if probabilities:
        english_probability = max(
            (language.prob for language in probabilities if language.lang == "en"),
            default=0.0,
        )
        return english_probability >= min_probability

    stopword_hits = sum(1 for token in alphabetic_tokens if token in ENGLISH_STOPWORD_FALLBACK)
    return (stopword_hits / max(len(alphabetic_tokens), 1)) >= 0.04

def sentiment_contributions(text, limit=30):
    if not text or not text.strip():
        return []

    normalized_text = re.sub(r"\s+", " ", text).strip()
    normalized_text_lower = normalized_text.lower()
    tokens = WORD_PATTERN.findall(normalized_text_lower)
    token_count = max(len(tokens), 1)
    length_scale = np.sqrt(token_count / 25)
    term_counts = {}
    contributions = {}

    for pattern, score in NEWS_PHRASE_LEXICON.items():
        occurrences = len(re.findall(pattern, normalized_text_lower))
        for _ in range(occurrences):
            weight = repeated_term_weight(term_counts, f"phrase:{pattern}")
            add_sentiment_contribution(
                contributions,
                clean_phrase_pattern(pattern),
                "phrase",
                score,
                weight,
            )

    for index, token in enumerate(tokens):
        previous_tokens = tokens[max(0, index - 3):index]
        modifiers = []
        if token in OBJECTIVE_CUES:
            objective_weight = repeated_term_weight(term_counts, f"objective:{token}")
            add_subjectivity_contribution(
                contributions,
                token,
                "objectif",
                -OBJECTIVE_SUBJECTIVITY_WEIGHT * SUBJECTIVITY_OBJECTIVE_BALANCE / length_scale,
                objective_weight,
            )
        if token in SUBJECTIVE_CUES:
            subjective_weight = repeated_term_weight(term_counts, f"subjective:{token}")
            add_subjectivity_contribution(
                contributions,
                token,
                "subjectif",
                SUBJECTIVE_SUBJECTIVITY_WEIGHT / length_scale,
                subjective_weight,
            )

        if token not in SENTIMENT_LEXICON:
            continue

        score = SENTIMENT_LEXICON[token]

        if any(previous in NEGATIONS for previous in previous_tokens):
            score *= -0.7
            modifiers.append("negation")

        for previous in previous_tokens:
            if previous in INTENSIFIERS:
                score *= INTENSIFIERS[previous]
                modifiers.append(previous)
            if previous in DIMINISHERS:
                score *= DIMINISHERS[previous]
                modifiers.append(previous)

        weight = repeated_term_weight(term_counts, f"word:{token}")
        add_sentiment_contribution(contributions, token, "mot", score, weight, modifiers)
        add_subjectivity_contribution(
            contributions,
            token,
            "mot",
            SUBJECTIVE_SUBJECTIVITY_WEIGHT * SENTIMENT_HIT_SIGNAL_WEIGHT / length_scale,
            weight,
            modifiers,
            count_occurrence=False,
        )

    rows = []
    for entry in contributions.values():
        occurrences = max(entry["occurrences"], 1)
        polarity_contribution = entry["contribution"]
        subjectivity_contribution = entry.get("subjectivity_contribution", 0.0)
        rows.append({
            "term": entry["term"],
            "kind": entry["kind"],
            "occurrences": entry["occurrences"],
            "total_weight": entry["total_weight"],
            "average_weight": entry["total_weight"] / occurrences,
            "average_score": entry["base_score_total"] / occurrences,
            "contribution": polarity_contribution,
            "polarity_contribution": polarity_contribution,
            "subjectivity_contribution": subjectivity_contribution,
            "impact": abs(polarity_contribution) + abs(subjectivity_contribution),
            "modifiers": sorted(entry["modifiers"]),
        })

    return sorted(rows, key=lambda row: row["impact"], reverse=True)[:limit]

def article_word_score_annotations(text):
    if not text:
        return []

    normalized_text = re.sub(r"\s+", " ", text).strip()
    tokens = WORD_PATTERN.findall(normalized_text.lower())
    token_count = max(len(tokens), 1)
    length_scale = np.sqrt(token_count / 25)
    term_counts = {}
    previous_tokens = []
    annotations = []

    for segment in re.findall(r"\s+|\S+", normalized_text):
        if segment.isspace():
            annotations.append({"text": segment, "space": True})
            continue

        token_match = WORD_PATTERN.search(segment.lower())
        token = token_match.group(0) if token_match else ""
        polarity_contribution = 0.0
        subjectivity_contribution = 0.0
        modifiers = []

        if token in OBJECTIVE_CUES:
            objective_weight = repeated_term_weight(term_counts, f"objective:{token}")
            subjectivity_contribution -= (
                OBJECTIVE_SUBJECTIVITY_WEIGHT
                * SUBJECTIVITY_OBJECTIVE_BALANCE
                * objective_weight
                / length_scale
            )

        if token in SUBJECTIVE_CUES:
            subjective_weight = repeated_term_weight(term_counts, f"subjective:{token}")
            subjectivity_contribution += (
                SUBJECTIVE_SUBJECTIVITY_WEIGHT
                * subjective_weight
                / length_scale
            )

        if token in SENTIMENT_LEXICON:
            score = SENTIMENT_LEXICON[token]
            context_tokens = previous_tokens[-3:]
            if any(previous in NEGATIONS for previous in context_tokens):
                score *= -0.7
                modifiers.append("negation")
            for previous in context_tokens:
                if previous in INTENSIFIERS:
                    score *= INTENSIFIERS[previous]
                    modifiers.append(previous)
                if previous in DIMINISHERS:
                    score *= DIMINISHERS[previous]
                    modifiers.append(previous)

            sentiment_weight = repeated_term_weight(term_counts, f"word:{token}")
            polarity_contribution = score * sentiment_weight
            subjectivity_contribution += (
                SUBJECTIVE_SUBJECTIVITY_WEIGHT
                * SENTIMENT_HIT_SIGNAL_WEIGHT
                * sentiment_weight
                / length_scale
            )

        annotations.append({
            "text": segment,
            "space": False,
            "token": token,
            "polarity": polarity_contribution,
            "subjectivity": subjectivity_contribution,
            "modifiers": modifiers,
            "has_score": abs(polarity_contribution) > 0 or abs(subjectivity_contribution) > 0,
        })

        if token:
            previous_tokens.append(token)

    return annotations

def read_article(article_html):
    title_tag = article_html.find('h1')
    title = fix_text_encoding(title_tag.get_text(" ", strip=True)) if title_tag else ""
    texts = article_html.find_all('p', class_='paragraph') or article_html.find_all('p')
    texts = fix_text_encoding(" ".join(text.get_text(" ", strip=True) for text in texts))
    return title, texts

def read_generic_article(article_html):
    title_tag = (
        article_html.find("h1")
        or article_html.find("meta", property="og:title")
        or article_html.find("title")
    )
    if title_tag and title_tag.name == "meta":
        title = title_tag.get("content", "")
    else:
        title = title_tag.get_text(" ", strip=True) if title_tag else ""

    article_node = article_html.find("article") or article_html
    paragraphs = article_node.find_all("p") or article_html.find_all("p")
    content = " ".join(paragraph.get_text(" ", strip=True) for paragraph in paragraphs)
    return fix_text_encoding(title), fix_text_encoding(content)

def save_article(title, topic, content, src, pol_score, sbj_score, published_date=None, publisher="CNN"):
    return Article.objects.create(
        title=fix_text_encoding(title),
        publisher=publisher,
        topic=topic,
        content=fix_text_encoding(content),
        source=src,
        polarity=pol_score,
        subjectivity=sbj_score,
        published_date=published_date,
    )

def preprocess_text(text):
    if PUNKT_AVAILABLE:
        tokens = word_tokenize(text.lower())
    else:
        tokens = WORD_PATTERN.findall(text.lower())
    
    stop_words = set(stopwords.words('english')) if STOPWORDS_AVAILABLE else set()
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def analyze_subjectivity(
    normalized_text,
    normalized_text_lower,
    tokens,
    blob_subjectivity,
    sentiment_hits,
    custom_polarity,
):
    token_count = max(len(tokens), 1)
    length_scale = np.sqrt(token_count / 25)
    objective_word_weight = weighted_token_count(tokens, OBJECTIVE_CUES, "objective")
    subjective_word_weight = weighted_token_count(tokens, SUBJECTIVE_CUES, "subjective")
    objective_phrase_weight = weighted_phrase_count(
        normalized_text_lower,
        OBJECTIVE_PHRASES,
        "objective-phrase",
    )
    subjective_phrase_weight = weighted_phrase_count(
        normalized_text_lower,
        SUBJECTIVE_PHRASES,
        "subjective-phrase",
    )
    evidence_weight = min(len(QUOTE_OR_NUMBER_PATTERN.findall(normalized_text)), 18)

    objective_signal = (
        objective_word_weight
        + OBJECTIVE_PHRASE_SIGNAL_WEIGHT * objective_phrase_weight
        + EVIDENCE_SIGNAL_WEIGHT * evidence_weight
    ) / length_scale
    subjective_signal = (
        subjective_word_weight
        + SUBJECTIVE_PHRASE_SIGNAL_WEIGHT * subjective_phrase_weight
        + SENTIMENT_HIT_SIGNAL_WEIGHT * sentiment_hits
        + POLARITY_SIGNAL_WEIGHT * abs(custom_polarity)
    ) / length_scale

    objective_signal = min(objective_signal, 1.6)
    subjective_signal = min(subjective_signal, 1.6)
    balanced_objective_signal = min(objective_signal * SUBJECTIVITY_OBJECTIVE_BALANCE, 1.6)
    cue_subjectivity = np.clip(
        0.24
        + SUBJECTIVE_SUBJECTIVITY_WEIGHT * subjective_signal
        - OBJECTIVE_SUBJECTIVITY_WEIGHT * balanced_objective_signal,
        0,
        1,
    )
    sentiment_density = min(sentiment_hits / token_count, 0.16)
    subjectivity_floor = 0.04 + sentiment_density
    subjectivity_ceiling = 1.0

    if objective_signal > 0.65 and subjective_signal < 0.35:
        subjectivity_ceiling = 0.42
    elif objective_signal > 1.0 and subjective_signal < 0.6:
        subjectivity_ceiling = 0.48

    return np.clip(
        max(0.42 * blob_subjectivity + 0.58 * cue_subjectivity, subjectivity_floor),
        0,
        subjectivity_ceiling,
    )

def analyze_sentiment(text):
    if not text or not text.strip():
        return 0.0, 0.0

    normalized_text = re.sub(r"\s+", " ", text).strip()
    normalized_text_lower = normalized_text.lower()
    tokens = WORD_PATTERN.findall(normalized_text.lower())
    blob_polarity = 0.0
    blob_subjectivity = 0.0
    vader_polarity = 0.0

    if TextBlob is not None:
        blob_sentiment = TextBlob(normalized_text).sentiment
        blob_polarity = blob_sentiment.polarity
        blob_subjectivity = blob_sentiment.subjectivity

    if VADER_ANALYZER is not None:
        vader_polarity = VADER_ANALYZER.polarity_scores(normalized_text)["compound"]

    custom_score = 0.0
    sentiment_hits = 0
    term_counts = {}

    for pattern, score in NEWS_PHRASE_LEXICON.items():
        occurrences = len(re.findall(pattern, normalized_text_lower))
        for _ in range(occurrences):
            weight = repeated_term_weight(term_counts, f"phrase:{pattern}")
            custom_score += score * weight
            sentiment_hits += weight

    for index, token in enumerate(tokens):
        if token not in SENTIMENT_LEXICON:
            continue

        score = SENTIMENT_LEXICON[token]
        previous_tokens = tokens[max(0, index - 3):index]

        if any(previous in NEGATIONS for previous in previous_tokens):
            score *= -0.7

        for previous in previous_tokens:
            score *= INTENSIFIERS.get(previous, 1.0)
            score *= DIMINISHERS.get(previous, 1.0)

        weight = repeated_term_weight(term_counts, f"word:{token}")
        custom_score += score * weight
        sentiment_hits += weight

    if sentiment_hits:
        custom_polarity = np.tanh(custom_score / np.sqrt(max(sentiment_hits, 1)))
    else:
        custom_polarity = 0.0

    sentiment_polarity = np.clip(
        (
            0.30 * blob_polarity
            + 0.30 * vader_polarity
            + 0.40 * custom_polarity
        ),
        -1,
        1
    )

    sentiment_subjectivity = analyze_subjectivity(
        normalized_text,
        normalized_text_lower,
        tokens,
        blob_subjectivity,
        sentiment_hits,
        custom_polarity,
    )

    return float(sentiment_polarity), float(sentiment_subjectivity)

def update_article_scores(article, force=False):
    if not force and article.polarity != 0 and article.subjectivity != 0:
        return

    article.polarity, article.subjectivity = analyze_sentiment(
        f"{article.title}. {article.content}"
    )
    article.save(update_fields=["polarity", "subjectivity"])

def normalize_cnn_url(link, base_url=CNN_BASE_URL):
    if not link:
        return ""
    link = link.strip()
    if link.startswith("//"):
        return f"https:{link}"
    if link.startswith("http://") or link.startswith("https://"):
        return link
    return urljoin(base_url, link.lstrip("/"))

def article_date_from_url(url):
    match = ARTICLE_DATE_PATTERN.search(url) or LOOSE_ARTICLE_DATE_PATTERN.search(url)
    if not match:
        return None
    try:
        return date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError:
        return None

def topic_from_url(url):
    path = re.sub(r"^https?://[^/]+/", "", url)
    path = re.sub(LINK_DATE_REGEX, "", path)
    topic = path.split("/")[0].strip()
    return topic or "general"

def domain_from_url(url):
    hostname = urlparse(url).hostname or ""
    hostname = hostname.removeprefix("www.")
    return hostname.split(".")[0] or "general"

def parse_iso_like_date(value):
    if not value:
        return None
    value = value.strip()
    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        return None

def source_topic_from_url(url, publisher=""):
    path_parts = [
        part
        for part in (urlparse(url).path or "").strip("/").split("/")
        if part
    ]
    if not path_parts:
        return domain_from_url(url)

    if publisher == "Al Jazeera":
        return path_parts[0]
    if publisher == "BBC":
        if path_parts[0] == "news":
            return "news"
        return path_parts[0]
    if publisher == "NPR":
        return "news"
    if publisher == "The Guardian":
        return path_parts[0]

    return topic_from_url(url)

def fetch_and_store_article(url, topic=None, published_date=None, publisher="CNN", fallback_title="", fallback_content=""):
    full_link = normalize_cnn_url(url)
    if not full_link:
        return {"seen": 0, "created": 0, "updated": 0, "skipped": 1, "url": url}

    published_date = published_date or article_date_from_url(full_link)
    topic = topic or source_topic_from_url(full_link, publisher)
    articles_match = Article.objects.filter(source=full_link)

    if articles_match.exists():
        article = articles_match.first()
        update_fields = []
        if publisher and article.publisher != publisher:
            article.publisher = publisher
            update_fields.append("publisher")
        if published_date and article.published_date != published_date:
            article.published_date = published_date
            update_fields.append("published_date")
        if topic and article.topic != topic:
            article.topic = topic
            update_fields.append("topic")
        update_article_scores(article)
        if update_fields:
            article.save(update_fields=update_fields)
        return {"seen": 1, "created": 0, "updated": 1, "skipped": 0, "url": full_link, "title": article.title}

    try:
        article_html = retrieve_webpage(full_link)
        if publisher == "CNN":
            title, content = read_article(article_html)
        else:
            title, content = read_generic_article(article_html)
    except requests.RequestException:
        title, content = fallback_title, fallback_content

    title = title or fallback_title
    content = content or fallback_content
    if not title:
        return {"seen": 1, "created": 0, "updated": 0, "skipped": 1, "url": full_link}
    if not is_english_text(f"{title}. {content}"):
        return {
            "seen": 1,
            "created": 0,
            "updated": 0,
            "skipped": 1,
            "url": full_link,
            "title": title,
            "reason": "non-english",
        }

    polarity_score, subjectivity_score = analyze_sentiment(f"{title}. {content}")
    save_article(
        title,
        topic,
        content,
        full_link,
        polarity_score,
        subjectivity_score,
        published_date=published_date,
        publisher=publisher,
    )
    return {"seen": 1, "created": 1, "updated": 0, "skipped": 0, "url": full_link, "title": title}

def empty_fetch_stats(target_date=None):
    return {
        "target_date": target_date,
        "seen": 0,
        "created": 0,
        "updated": 0,
        "skipped": 0,
        "errors": 0,
        "discovered": 0,
        "sitemaps": 0,
        "last_url": "",
    }

def add_fetch_result(stats, result):
    for key in ["seen", "created", "updated", "skipped"]:
        stats[key] += result.get(key, 0)
    stats["last_url"] = result.get("url", stats.get("last_url", ""))
    return stats

def child_text(element, tag_name):
    for child in element:
        if child.tag.endswith(tag_name):
            return child.text.strip() if child.text else ""
    return ""

def parse_feed_date(value):
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError, IndexError, OverflowError):
        return None

def rss_items(feed_content):
    root = ElementTree.fromstring(feed_content)
    items = [
        element
        for element in root.iter()
        if element.tag.endswith("item") or element.tag.endswith("entry")
    ]
    rows = []
    for item in items:
        link = child_text(item, "link")
        if not link:
            link_node = next((child for child in item if child.tag.endswith("link")), None)
            link = link_node.get("href", "") if link_node is not None else ""
        summary = child_text(item, "description") or child_text(item, "summary")
        rows.append({
            "title": fix_text_encoding(child_text(item, "title")),
            "url": normalize_cnn_url(link),
            "published_date": parse_feed_date(
                child_text(item, "pubDate")
                or child_text(item, "published")
                or child_text(item, "updated")
            ),
            "topic": fix_text_encoding(child_text(item, "category")),
            "summary": fix_text_encoding(BeautifulSoup(summary, "html.parser").get_text(" ", strip=True)),
        })
    return [row for row in rows if row["url"] and row["title"]]

def xml_locations(xml_content):
    root = ElementTree.fromstring(xml_content)
    return [
        element.text.strip()
        for element in root.iter()
        if element.tag.endswith("loc") and element.text and element.text.strip()
    ]

def sitemap_url_rows(xml_content):
    root = ElementTree.fromstring(xml_content)
    rows = []
    for url_element in root.iter():
        if not url_element.tag.endswith("url"):
            continue

        row = {"url": "", "lastmod": None, "published_date": None, "title": ""}
        for child in url_element:
            child_name = child.tag.split("}")[-1]
            if child_name == "loc" and child.text and not row["url"]:
                row["url"] = normalize_cnn_url(child.text.strip())
            elif child_name == "lastmod" and child.text:
                row["lastmod"] = parse_iso_like_date(child.text)
            elif child_name == "news":
                for news_child in child.iter():
                    news_child_name = news_child.tag.split("}")[-1]
                    if news_child_name == "publication_date" and news_child.text:
                        row["published_date"] = parse_iso_like_date(news_child.text)
                    elif news_child_name == "title" and news_child.text:
                        row["title"] = fix_text_encoding(news_child.text)

        if row["url"]:
            rows.append(row)
    return rows

def sitemap_index_rows(xml_content):
    root = ElementTree.fromstring(xml_content)
    rows = []
    for sitemap_element in root.iter():
        if not sitemap_element.tag.endswith("sitemap"):
            continue

        row = {"url": "", "lastmod": None}
        for child in sitemap_element:
            child_name = child.tag.split("}")[-1]
            if child_name == "loc" and child.text:
                row["url"] = normalize_cnn_url(child.text.strip())
            elif child_name == "lastmod" and child.text:
                row["lastmod"] = parse_iso_like_date(child.text)

        if row["url"]:
            rows.append(row)
    return rows

@lru_cache(maxsize=32)
def sitemap_index_rows_from_url(sitemap_url):
    response = requests.get(sitemap_url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    return tuple(
        (row["url"], row["lastmod"])
        for row in sitemap_index_rows(response.content)
    )

@lru_cache(maxsize=160)
def article_rows_from_sitemap(sitemap_url):
    response = requests.get(sitemap_url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    return tuple(
        (
            row["url"],
            row["published_date"],
            row["lastmod"],
            row["title"],
        )
        for row in sitemap_url_rows(response.content)
    )

def historical_article_item(url, publisher, published_date=None, title=""):
    return {
        "url": normalize_cnn_url(url),
        "publisher": publisher,
        "published_date": published_date or article_date_from_url(url),
        "topic": source_topic_from_url(url, publisher),
        "title": fix_text_encoding(title),
        "summary": "",
    }

def is_historical_article_url(url, publisher):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ""
    path = parsed_url.path or ""

    if "/video/" in path or "/videos/" in path:
        return False
    if publisher == "Al Jazeera":
        return hostname.endswith("aljazeera.com") and "/wp-content/" not in path
    if publisher == "NPR":
        return hostname == "www.npr.org" and article_date_from_url(url) is not None
    if publisher == "BBC":
        return hostname == "www.bbc.com" and path.startswith("/news/")
    return True

def article_items_from_sitemaps(sitemap_urls, publisher, target_date, log_callback=None):
    items = {}
    for sitemap_index, sitemap_url in enumerate(sitemap_urls, start=1):
        try:
            if log_callback:
                log_callback(f"{publisher}: reading sitemap {sitemap_index}/{len(sitemap_urls)}: {sitemap_url}")
            rows = article_rows_from_sitemap(sitemap_url)
        except (requests.RequestException, ElementTree.ParseError):
            if log_callback:
                log_callback(f"{publisher}: could not read sitemap: {sitemap_url}", "warning")
            continue

        matched_before = len(items)
        for url, row_published_date, row_lastmod, title in rows:
            item_date = row_published_date or article_date_from_url(url) or row_lastmod
            if item_date != target_date:
                continue
            if not is_historical_article_url(url, publisher):
                continue
            items[url] = historical_article_item(
                url,
                publisher,
                published_date=item_date,
                title=title,
            )

        if log_callback:
            log_callback(
                f"{publisher}: sitemap {sitemap_index}/{len(sitemap_urls)} matched "
                f"{len(items) - matched_before} articles for {target_date.isoformat()}."
            )

    return list(items.values())

@lru_cache(maxsize=1)
def cnn_article_sitemap_index_urls():
    response = requests.get(
        CNN_ARTICLE_SITEMAP_INDEX_URL,
        headers=REQUEST_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return tuple(xml_locations(response.content))

def cnn_sitemap_urls_for_month(target_date):
    month_path = f"/{target_date:%Y/%m}.xml"
    return [
        sitemap_url
        for sitemap_url in cnn_article_sitemap_index_urls()
        if sitemap_url.endswith(month_path)
    ]

@lru_cache(maxsize=256)
def cnn_article_urls_from_sitemap(sitemap_url):
    response = requests.get(sitemap_url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    return tuple(xml_locations(response.content))

def discover_cnn_article_urls_for_date(target_date, log_callback=None):
    urls = set()
    target_path = f"/{target_date:%Y/%m/%d}/"
    if log_callback:
        log_callback(f"Loading CNN article sitemap index for {target_date:%Y-%m}.")
    sitemap_urls = cnn_sitemap_urls_for_month(target_date)
    if log_callback:
        log_callback(f"Found {len(sitemap_urls)} monthly section sitemaps for {target_date:%Y-%m}.")

    for sitemap_index, sitemap_url in enumerate(sitemap_urls, start=1):
        try:
            if log_callback:
                log_callback(f"Reading sitemap {sitemap_index}/{len(sitemap_urls)}: {sitemap_url}")
            sitemap_article_urls = cnn_article_urls_from_sitemap(sitemap_url)
        except (requests.RequestException, ElementTree.ParseError):
            if log_callback:
                log_callback(f"Could not read sitemap: {sitemap_url}", "warning")
            continue

        matched_before = len(urls)
        for loc in sitemap_article_urls:
            if target_path in loc:
                urls.add(loc)
        if log_callback:
            log_callback(
                f"Sitemap {sitemap_index}/{len(sitemap_urls)} matched {len(urls) - matched_before} articles for {target_date.isoformat()}."
            )

    return sorted(
        url
        for url in urls
        if url and "/video/" not in url and "/videos/" not in url
    )

def retrieve_cnn_articles_for_date(target_date, limit=None, log_callback=None):
    stats = empty_fetch_stats(target_date)
    article_urls = discover_cnn_article_urls_for_date(target_date, log_callback=log_callback)
    stats["discovered"] = len(article_urls)
    stats["sitemaps"] = len(cnn_sitemap_urls_for_month(target_date))
    if log_callback:
        log_callback(f"Discovered {stats['discovered']} article URLs for {target_date.isoformat()}.")

    if limit is not None:
        article_urls = article_urls[:limit]
        if log_callback:
            log_callback(f"Applying per-day limit: processing {len(article_urls)} article URLs.")

    for article_index, article_url in enumerate(article_urls, start=1):
        try:
            if log_callback:
                log_callback(f"Fetching article {article_index}/{len(article_urls)}: {article_url}")
            result = fetch_and_store_article(
                article_url,
                topic=topic_from_url(article_url),
                published_date=target_date,
            )
            add_fetch_result(stats, result)
            if log_callback:
                if result.get("created"):
                    action = "created"
                elif result.get("updated"):
                    action = "updated"
                elif result.get("skipped"):
                    action = "skipped"
                else:
                    action = "seen"
                title = result.get("title") or article_url
                log_callback(f"Article {article_index}/{len(article_urls)} {action}: {title}")
        except Exception as exc:
            stats["errors"] += 1
            stats["last_url"] = article_url
            if log_callback:
                log_callback(f"Article {article_index}/{len(article_urls)} failed: {article_url} ({exc})", "error")
            print(f"Failed to fetch {article_url}: {exc}")

    return stats

def al_jazeera_sitemap_urls_for_date(target_date):
    urls = []
    daily_sitemap = f"https://www.aljazeera.com/sitemaps/article-new/{target_date:%d-%m-%Y}.xml"
    monthly_sitemap = f"https://www.aljazeera.com/sitemaps/article-archive/{target_date:%Y/%m}.xml"
    known_sitemaps = {
        url
        for url, _ in (
            list(sitemap_index_rows_from_url(AL_JAZEERA_ARTICLE_NEW_INDEX_URL))
            + list(sitemap_index_rows_from_url(AL_JAZEERA_ARTICLE_ARCHIVE_INDEX_URL))
        )
    }

    if daily_sitemap in known_sitemaps:
        urls.append(daily_sitemap)
    if monthly_sitemap in known_sitemaps:
        urls.append(monthly_sitemap)
    return urls

def npr_sitemap_urls_for_date(target_date):
    half = "Jan" if target_date.month <= 6 else "Jul"
    expected_suffix = f"sitemap_standard_01-{half}-{target_date:%y}.xml"
    sitemap_urls = [
        url
        for url, _ in sitemap_index_rows_from_url(NPR_STANDARD_INDEX_URL)
        if url.endswith(expected_suffix)
    ]
    if (date.today() - target_date).days <= 3:
        sitemap_urls.append(NPR_NEWS_SITEMAP_URL)
    return sitemap_urls

def guardian_api_items_for_date(target_date, log_callback=None):
    items = []
    page = 1
    page_size = 50
    total_pages = 1

    while page <= total_pages:
        response = requests.get(
            GUARDIAN_SEARCH_API_URL,
            headers=REQUEST_HEADERS,
            params={
                "api-key": GUARDIAN_API_KEY,
                "from-date": target_date.isoformat(),
                "to-date": target_date.isoformat(),
                "page": page,
                "page-size": page_size,
                "show-fields": "trailText,bodyText",
                "order-by": "newest",
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json().get("response", {})
        total_pages = min(int(payload.get("pages") or 1), 20)
        results = payload.get("results", [])

        if log_callback:
            log_callback(
                f"The Guardian: API page {page}/{total_pages} returned "
                f"{len(results)} article(s)."
            )

        for result in results:
            web_url = result.get("webUrl", "")
            published_date = parse_iso_like_date(result.get("webPublicationDate", ""))
            if not web_url or published_date != target_date:
                continue

            fields = result.get("fields") or {}
            trail_text = BeautifulSoup(
                fields.get("trailText", ""),
                "html.parser",
            ).get_text(" ", strip=True)
            body_text = fields.get("bodyText", "")
            items.append({
                "url": normalize_cnn_url(web_url),
                "publisher": "The Guardian",
                "published_date": published_date,
                "topic": result.get("sectionId") or source_topic_from_url(web_url, "The Guardian"),
                "title": fix_text_encoding(result.get("webTitle", "")),
                "summary": fix_text_encoding(body_text or trail_text),
            })

        page += 1

    return items

def bbc_sitemap_urls_for_date(target_date):
    urls = []

    recent_news_rows = list(sitemap_index_rows_from_url(BBC_NEWS_INDEX_URL))
    if target_date >= date.today().replace(day=1):
        urls.extend(url for url, _ in recent_news_rows)

    archive_rows = list(sitemap_index_rows_from_url(BBC_ARCHIVE_INDEX_URL))
    dated_rows = [
        (index, url, lastmod)
        for index, (url, lastmod) in enumerate(archive_rows)
        if lastmod
    ]
    selected_indexes = set()
    for index, _, lastmod in dated_rows:
        if lastmod >= target_date:
            selected_indexes.update({index - 1, index, index + 1})
            break
    if not selected_indexes and dated_rows:
        selected_indexes.update({dated_rows[-1][0] - 1, dated_rows[-1][0]})

    for index in sorted(selected_indexes):
        if 0 <= index < len(archive_rows):
            urls.append(archive_rows[index][0])

    return list(dict.fromkeys(urls))

def discover_historical_article_items_for_source(source_config, target_date, log_callback=None):
    publisher = source_config["name"]
    if source_config["kind"] == "al_jazeera":
        sitemap_urls = al_jazeera_sitemap_urls_for_date(target_date)
    elif source_config["kind"] == "npr":
        sitemap_urls = npr_sitemap_urls_for_date(target_date)
    elif source_config["kind"] == "guardian":
        items = guardian_api_items_for_date(target_date, log_callback=log_callback)
        if log_callback:
            log_callback(
                f"{publisher}: API matched {len(items)} articles for {target_date.isoformat()}."
            )
        return items
    elif source_config["kind"] == "bbc":
        sitemap_urls = bbc_sitemap_urls_for_date(target_date)
    else:
        sitemap_urls = []

    if log_callback:
        log_callback(
            f"{publisher}: selected {len(sitemap_urls)} historical sitemap(s) "
            f"for {target_date.isoformat()}."
        )
    return article_items_from_sitemaps(
        sitemap_urls,
        publisher,
        target_date,
        log_callback=log_callback,
    )

def retrieve_historical_articles_for_source(source_config, target_date, limit=None, log_callback=None):
    stats = empty_fetch_stats(target_date)
    stats["source"] = source_config["name"]
    article_items = discover_historical_article_items_for_source(
        source_config,
        target_date,
        log_callback=log_callback,
    )
    stats["discovered"] = len(article_items)

    if source_config["kind"] == "al_jazeera":
        stats["sitemaps"] = len(al_jazeera_sitemap_urls_for_date(target_date))
    elif source_config["kind"] == "npr":
        stats["sitemaps"] = len(npr_sitemap_urls_for_date(target_date))
    elif source_config["kind"] == "bbc":
        stats["sitemaps"] = len(bbc_sitemap_urls_for_date(target_date))
    elif source_config["kind"] == "guardian":
        stats["sitemaps"] = max((stats["discovered"] + 49) // 50, 1 if stats["discovered"] else 0)

    if limit is not None:
        article_items = article_items[:limit]
        if log_callback:
            log_callback(f"{source_config['name']}: applying limit, processing {len(article_items)} URLs.")

    if log_callback:
        log_callback(f"{source_config['name']}: discovered {stats['discovered']} article URLs.")

    for article_index, item in enumerate(article_items, start=1):
        try:
            if log_callback:
                log_callback(
                    f"Fetching {source_config['name']} article "
                    f"{article_index}/{len(article_items)}: {item['url']}"
                )
            result = fetch_and_store_article(
                item["url"],
                topic=item["topic"],
                published_date=item["published_date"],
                publisher=source_config["name"],
                fallback_title=item["title"],
                fallback_content=item["summary"],
            )
            add_fetch_result(stats, result)
            if log_callback:
                if result.get("created"):
                    action = "created"
                elif result.get("updated"):
                    action = "updated"
                elif result.get("skipped"):
                    action = "skipped"
                else:
                    action = "seen"
                log_callback(
                    f"{source_config['name']} article {article_index}/{len(article_items)} "
                    f"{action}: {result.get('title') or item['title'] or item['url']}"
                )
        except Exception as exc:
            stats["errors"] += 1
            stats["last_url"] = item["url"]
            if log_callback:
                log_callback(
                    f"{source_config['name']} article {article_index}/{len(article_items)} "
                    f"failed: {item['url']} ({exc})",
                    "error",
                )
            print(f"Failed to fetch {item['url']}: {exc}")

    return stats

def combine_fetch_stats(target_date, source_stats):
    stats = empty_fetch_stats(target_date)
    stats["sources"] = []
    for current_stats in source_stats:
        stats["sources"].append(current_stats.get("source") or "CNN")
        for key in ["seen", "created", "updated", "skipped", "errors", "discovered", "sitemaps"]:
            stats[key] += current_stats.get(key, 0)
        stats["last_url"] = current_stats.get("last_url") or stats.get("last_url", "")
    return stats

def retrieve_all_articles_for_date(target_date, limit_per_source=None, log_callback=None):
    source_stats = []
    if log_callback:
        log_callback(
            f"Starting historical-source retrieval for {target_date.isoformat()} "
            f"({len(HISTORICAL_NEWS_SOURCES)} sources)."
        )

    for source_config in HISTORICAL_NEWS_SOURCES:
        try:
            if source_config["kind"] == "cnn":
                current_stats = retrieve_cnn_articles_for_date(
                    target_date,
                    limit=limit_per_source,
                    log_callback=log_callback,
                )
                current_stats["source"] = "CNN"
            else:
                current_stats = retrieve_historical_articles_for_source(
                    source_config,
                    target_date,
                    limit=limit_per_source,
                    log_callback=log_callback,
                )
            source_stats.append(current_stats)
        except Exception as exc:
            current_stats = empty_fetch_stats(target_date)
            current_stats["source"] = source_config["name"]
            current_stats["errors"] = 1
            source_stats.append(current_stats)
            if log_callback:
                log_callback(f"{source_config['name']} retrieval failed: {exc}", "error")

    stats = combine_fetch_stats(target_date, source_stats)
    if log_callback:
        log_callback(
            f"Completed historical-source retrieval for {target_date.isoformat()}: "
            f"{stats['discovered']} discovered, {stats['created']} created, "
            f"{stats['updated']} updated, {stats['skipped']} skipped, {stats['errors']} errors."
        )
    return stats

def clear_source_discovery_caches():
    cnn_article_sitemap_index_urls.cache_clear()
    cnn_article_urls_from_sitemap.cache_clear()
    sitemap_index_rows_from_url.cache_clear()
    article_rows_from_sitemap.cache_clear()
