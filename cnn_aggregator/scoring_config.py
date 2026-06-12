SCORING_VERSION = "2026.06.12"

SUBJECTIVITY_OBJECTIVE_THRESHOLD = 0.30
SUBJECTIVITY_SUBJECTIVE_THRESHOLD = 0.50

LEXICON_SCORE_BOOST = 1.18
REPEATED_TERM_TOTAL_WEIGHT_CAP = 2.5

OBJECTIVE_PHRASE_SIGNAL_WEIGHT = 1.35
SUBJECTIVE_PHRASE_SIGNAL_WEIGHT = 1.45
EVIDENCE_SIGNAL_WEIGHT = 0.45
SENTIMENT_HIT_SIGNAL_WEIGHT = 0.40
POLARITY_SIGNAL_WEIGHT = 0.70
OBJECTIVE_SUBJECTIVITY_WEIGHT = 0.24
SUBJECTIVE_SUBJECTIVITY_WEIGHT = 0.36

QUOTE_SENTIMENT_WEIGHT = 0.45
QUOTE_SUBJECTIVITY_WEIGHT = 0.35
QUOTE_WRITING_TONE_WEIGHT = 0.25

DEFAULT_POLARITY_WEIGHTS = {
    "blob": 0.30,
    "vader": 0.30,
    "custom": 0.40,
}

FATAL_INCIDENT_POLARITY_WEIGHTS = {
    "blob": 0.15,
    "vader": 0.15,
    "custom": 0.70,
}

SUBJECTIVITY_WEIGHTS = {
    "blob": 0.42,
    "cue": 0.58,
}

SUBJECTIVITY_SIGNAL_CAP = 1.6

SUBJECTIVITY_TOPIC_PRIORS = {
    "opinion": 0.30,
    "entertainment": 0.16,
    "culture": 0.12,
    "lifestyle": 0.10,
    "sports": 0.06,
}

TOPIC_SENTIMENT_MULTIPLIERS = {
    "sports": {
        "champion": 0.55,
        "defeat": 0.35,
        "defeated": 0.35,
        "endurance": 0.25,
        "good": 0.60,
        "help": 0.50,
        "loss": 0.45,
        "record": 0.20,
        "strong": 0.45,
        "victory": 0.45,
        "win": 0.45,
        "winning": 0.45,
        "wins": 0.45,
    },
    "business": {
        "loss": 1.20,
        "losses": 1.20,
        "profit": 1.15,
        "revenue": 1.10,
    },
    "entertainment": {
        "awful": 1.20,
        "brilliant": 1.20,
        "excellent": 1.20,
        "good": 1.10,
        "horrible": 1.20,
        "wonderful": 1.20,
    },
}

TOPIC_POLARITY_MULTIPLIERS = {
    "sports": 0.78,
}
