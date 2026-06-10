from datetime import date, timedelta

from django.test import SimpleTestCase, TestCase

from cnn_aggregator.management.commands.run_cnn_worker import Command

from .models import Article
from .utils import analyze_sentiment, repeated_term_weight, sentiment_contributions


class SentimentAnalysisTests(SimpleTestCase):
    def test_repeated_term_weight_has_diminishing_returns(self):
        counts = {}

        weights = [repeated_term_weight(counts, "word:protected") for _ in range(4)]

        self.assertEqual(weights[0], 1)
        self.assertLess(weights[1], weights[0])
        self.assertLess(weights[2], weights[1])
        self.assertLess(weights[3], weights[2])

    def test_repeated_positive_word_does_not_dominate_polarity(self):
        single_score, _ = analyze_sentiment("protected")
        repeated_five_score, _ = analyze_sentiment("protected " * 5)
        repeated_twenty_score, _ = analyze_sentiment("protected " * 20)
        varied_score, _ = analyze_sentiment(
            "protected approved benefit breakthrough growth recovery secure successful victory agreement"
        )

        early_gain = repeated_five_score - single_score
        later_gain = repeated_twenty_score - repeated_five_score

        self.assertGreater(repeated_twenty_score, single_score)
        self.assertLess(later_gain, early_gain)
        self.assertGreater(varied_score, repeated_twenty_score)

    def test_sentiment_contributions_explain_repeated_terms(self):
        contributions = sentiment_contributions("protected protected protected")
        protected = next(item for item in contributions if item["term"] == "protected")

        self.assertEqual(protected["occurrences"], 3)
        self.assertLess(protected["total_weight"], 3)
        self.assertGreater(protected["contribution"], 0)


class WorkerFreshnessTests(TestCase):
    def test_freshness_dates_start_from_today_and_include_latest_article_date(self):
        latest_date = date.today() - timedelta(days=1)
        Article.objects.create(
            title="Known article",
            topic="world",
            content="Existing article content",
            polarity=0,
            subjectivity=0,
            source="https://example.com/known",
            published_date=latest_date,
        )

        dates, stored_latest_date = Command()._freshness_dates(window_days=3)

        self.assertEqual(stored_latest_date, latest_date)
        self.assertEqual(dates[:2], [date.today(), latest_date])

    def test_freshness_dates_are_capped_by_window(self):
        latest_date = date.today() - timedelta(days=10)
        Article.objects.create(
            title="Older known article",
            topic="world",
            content="Existing article content",
            polarity=0,
            subjectivity=0,
            source="https://example.com/older-known",
            published_date=latest_date,
        )

        dates, stored_latest_date = Command()._freshness_dates(window_days=3)

        self.assertEqual(stored_latest_date, latest_date)
        self.assertEqual(len(dates), 3)
        self.assertEqual(dates[0], date.today())
