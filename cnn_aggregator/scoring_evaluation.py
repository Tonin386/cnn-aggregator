SCORING_EVALUATION_CASES = [
    {
        "name": "sourced_factual_report",
        "topic": "politics",
        "publisher": "NPR",
        "text": (
            "Officials said the committee released a public report after reviewing "
            "court records, testimony and 42 documents."
        ),
        "expected": {
            "subjectivity_max": 0.30,
            "polarity_abs_max": 0.45,
        },
    },
    {
        "name": "explicit_opinion_language",
        "topic": "opinion",
        "publisher": "The Guardian",
        "text": (
            "This outrageous plan is clearly terrible and should be stopped, "
            "critics argue in a blistering column."
        ),
        "expected": {
            "subjectivity_min": 0.50,
            "polarity_max": -0.20,
        },
    },
    {
        "name": "quoted_opinion_attribution",
        "topic": "politics",
        "publisher": "CNN",
        "text": (
            "Officials said critics called the plan \"outrageous and terrible\" "
            "during a public hearing."
        ),
        "expected": {
            "editorial_subjectivity_max": 0.45,
            "writing_vs_event_abs_max": 0.15,
        },
    },
    {
        "name": "fatal_incident",
        "topic": "us",
        "publisher": "The Guardian",
        "text": (
            "Officials recovered the body of a missing child after a fatal incident. "
            "Rescue crews said the search lasted 30 hours."
        ),
        "expected": {
            "polarity_max": -0.20,
            "subjectivity_max": 0.50,
        },
    },
]
