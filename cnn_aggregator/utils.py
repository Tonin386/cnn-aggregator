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
from datetime import date
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin
from xml.etree import ElementTree

from .models import Article

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

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
QUOTE_OR_NUMBER_PATTERN = re.compile(r'["“”]|\b\d+(?:\.\d+)?%?\b')
WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?%?")
CNN_BASE_URL = "https://www.cnn.com/"
CNN_ARTICLE_SITEMAP_INDEX_URL = "https://www.cnn.com/sitemap/article.xml"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
}
ARTICLE_DATE_PATTERN = re.compile(r"/(?P<year>20\d{2})/(?P<month>\d{2})/(?P<day>\d{2})/")
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

def sentiment_contributions(text, limit=30):
    if not text or not text.strip():
        return []

    normalized_text = re.sub(r"\s+", " ", text).strip()
    normalized_text_lower = normalized_text.lower()
    tokens = WORD_PATTERN.findall(normalized_text_lower)
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
        if token not in SENTIMENT_LEXICON:
            continue

        score = SENTIMENT_LEXICON[token]
        previous_tokens = tokens[max(0, index - 3):index]
        modifiers = []

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

    rows = []
    for entry in contributions.values():
        occurrences = max(entry["occurrences"], 1)
        rows.append({
            "term": entry["term"],
            "kind": entry["kind"],
            "occurrences": entry["occurrences"],
            "total_weight": entry["total_weight"],
            "average_weight": entry["total_weight"] / occurrences,
            "average_score": entry["base_score_total"] / occurrences,
            "contribution": entry["contribution"],
            "modifiers": sorted(entry["modifiers"]),
        })

    return sorted(rows, key=lambda row: abs(row["contribution"]), reverse=True)[:limit]

def read_article(article_html):
    title_tag = article_html.find('h1')
    title = fix_text_encoding(title_tag.get_text(" ", strip=True)) if title_tag else ""
    texts = article_html.find_all('p', class_='paragraph') or article_html.find_all('p')
    texts = fix_text_encoding(" ".join(text.get_text(" ", strip=True) for text in texts))
    return title, texts

def save_article(title, topic, content, src, pol_score, sbj_score, published_date=None):
    return Article.objects.create(
        title=fix_text_encoding(title),
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

    objective_cue_count = sum(token in OBJECTIVE_CUES for token in tokens)
    subjective_cue_count = sum(token in SUBJECTIVE_CUES for token in tokens)
    evidence_count = len(QUOTE_OR_NUMBER_PATTERN.findall(normalized_text))
    token_count = max(len(tokens), 1)

    subjectivity_adjustment = (
        0.04 * subjective_cue_count
        - 0.035 * objective_cue_count
        - 0.015 * evidence_count
    ) / np.sqrt(token_count / 25)

    sentiment_density = min(sentiment_hits / token_count, 0.2)
    subjectivity_floor = 0.06 + sentiment_density

    sentiment_subjectivity = np.clip(
        max(blob_subjectivity + subjectivity_adjustment, subjectivity_floor),
        0,
        1
    )

    return float(sentiment_polarity), float(sentiment_subjectivity)

def update_article_scores(article):
    if article.polarity != 0 and article.subjectivity != 0:
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
    match = ARTICLE_DATE_PATTERN.search(url)
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

def fetch_and_store_article(url, topic=None, published_date=None):
    full_link = normalize_cnn_url(url)
    if not full_link:
        return {"seen": 0, "created": 0, "updated": 0, "skipped": 1, "url": url}

    published_date = published_date or article_date_from_url(full_link)
    topic = topic or topic_from_url(full_link)
    articles_match = Article.objects.filter(source=full_link)

    if articles_match.exists():
        article = articles_match.first()
        update_fields = []
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

    title, content = read_article(retrieve_webpage(full_link))
    if not title:
        return {"seen": 1, "created": 0, "updated": 0, "skipped": 1, "url": full_link}

    polarity_score, subjectivity_score = analyze_sentiment(f"{title}. {content}")
    save_article(
        title,
        topic,
        content,
        full_link,
        polarity_score,
        subjectivity_score,
        published_date=published_date,
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

def xml_locations(xml_content):
    root = ElementTree.fromstring(xml_content)
    return [
        element.text.strip()
        for element in root.iter()
        if element.tag.endswith("loc") and element.text and element.text.strip()
    ]

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
