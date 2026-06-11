from datetime import date, timedelta
from unittest.mock import patch

from django.test import Client, SimpleTestCase, TestCase

from cnn_aggregator.management.commands.run_cnn_worker import Command

from .models import Article, WorkerState
from .utils import (
    OBJECTIVE_CUES,
    OBJECTIVE_PHRASE_SIGNAL_WEIGHT,
    OBJECTIVE_PHRASES,
    OBJECTIVE_SUBJECTIVITY_WEIGHT,
    SUBJECTIVE_CUES,
    SUBJECTIVE_PHRASE_SIGNAL_WEIGHT,
    SUBJECTIVE_PHRASES,
    SUBJECTIVE_SUBJECTIVITY_WEIGHT,
    SUBJECTIVITY_OBJECTIVE_BALANCE,
    HISTORICAL_NEWS_SOURCES,
    analyze_sentiment,
    article_word_score_annotations,
    classify_subjectivity,
    repeated_term_weight,
    retrieve_all_articles_for_date,
    sentiment_contributions,
    rss_items,
)
from .views import apply_article_filters, build_word_cloud_visual, word_cloud_rows


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
        self.assertIn("subjectivity_contribution", protected)

    def test_article_word_annotations_include_real_polarity_and_subjectivity_scores(self):
        annotations = article_word_score_annotations("Officials said progress was protected.")
        scored_words = {item["token"]: item for item in annotations if item.get("has_score")}

        self.assertIn("officials", scored_words)
        self.assertIn("progress", scored_words)
        self.assertLess(scored_words["officials"]["subjectivity"], 0)
        self.assertGreater(scored_words["progress"]["polarity"], 0)

    def test_subjectivity_is_lower_for_sourced_factual_copy(self):
        _, factual_subjectivity = analyze_sentiment(
            "According to court documents, officials said 42 percent of records "
            "were reviewed in a public report. Police said witnesses testified in a statement."
        )
        _, opinion_subjectivity = analyze_sentiment(
            "Critics say the plan is outrageous and clearly terrible. It might have "
            "devastating consequences and should have been stopped, some believe."
        )

        self.assertLess(factual_subjectivity, 0.25)
        self.assertGreater(opinion_subjectivity, 0.55)
        self.assertGreater(opinion_subjectivity, factual_subjectivity)

    def test_subjectivity_classification_uses_model_calibrated_thresholds(self):
        self.assertEqual(classify_subjectivity(0.2999), "Objectif")
        self.assertEqual(classify_subjectivity(0.30), "Insuffisamment objectif")
        self.assertEqual(classify_subjectivity(0.4999), "Insuffisamment objectif")
        self.assertEqual(classify_subjectivity(0.50), "Subjectif")

    def test_subjectivity_lexicon_masses_are_balanced(self):
        objective_mass = OBJECTIVE_SUBJECTIVITY_WEIGHT * SUBJECTIVITY_OBJECTIVE_BALANCE * (
            len(OBJECTIVE_CUES) + OBJECTIVE_PHRASE_SIGNAL_WEIGHT * len(OBJECTIVE_PHRASES)
        )
        subjective_mass = SUBJECTIVE_SUBJECTIVITY_WEIGHT * (
            len(SUBJECTIVE_CUES) + SUBJECTIVE_PHRASE_SIGNAL_WEIGHT * len(SUBJECTIVE_PHRASES)
        )

        self.assertAlmostEqual(objective_mass, subjective_mass)


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

    def test_worker_rescores_existing_articles(self):
        article = Article.objects.create(
            title="Outrageous plan",
            topic="world",
            content="Critics say the plan is outrageous and clearly terrible.",
            polarity=0,
            subjectivity=0,
            source="https://example.com/rescore",
            published_date=date.today(),
        )
        state = WorkerState.get_solo()

        Command()._rescore_existing_articles(state)

        article.refresh_from_db()
        self.assertNotEqual(article.polarity, 0)
        self.assertGreater(article.subjectivity, 0)

    @patch("cnn_aggregator.views.subprocess.Popen")
    def test_dashboard_can_start_rescore_only_worker(self, popen):
        response = Client(SERVER_NAME="localhost").post("/worker/rescore/")

        self.assertEqual(response.status_code, 200)
        command = popen.call_args.args[0]
        self.assertIn("--rescore-existing", command)
        self.assertIn("--once", command)

    @patch("cnn_aggregator.views.subprocess.Popen")
    def test_dashboard_can_start_rescore_then_continue_worker(self, popen):
        response = Client(SERVER_NAME="localhost").post("/worker/rescore/?continue=1")

        self.assertEqual(response.status_code, 200)
        command = popen.call_args.args[0]
        self.assertIn("--rescore-existing", command)
        self.assertNotIn("--once", command)

    @patch("cnn_aggregator.utils.retrieve_historical_articles_for_source")
    @patch("cnn_aggregator.utils.retrieve_cnn_articles_for_date")
    def test_all_source_retrieval_runs_each_historical_source_for_the_same_date(self, cnn_fetch, historical_fetch):
        target_date = date.today()
        cnn_fetch.return_value = {
            "source": "CNN",
            "target_date": target_date,
            "seen": 1,
            "created": 1,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "discovered": 1,
            "sitemaps": 1,
            "last_url": "https://cnn.example/article",
        }
        historical_fetch.side_effect = lambda source_config, target_date, limit=None, log_callback=None: {
            "source": source_config["name"],
            "target_date": target_date,
            "seen": 1,
            "created": 1,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "discovered": 1,
            "sitemaps": 1,
            "last_url": f"https://{source_config['name'].lower().replace(' ', '-')}.example/article",
        }

        stats = retrieve_all_articles_for_date(target_date, limit_per_source=5)

        cnn_fetch.assert_called_once_with(target_date, limit=5, log_callback=None)
        self.assertEqual(historical_fetch.call_count, len(HISTORICAL_NEWS_SOURCES) - 1)
        self.assertEqual(stats["created"], len(HISTORICAL_NEWS_SOURCES))
        self.assertEqual(stats["discovered"], len(HISTORICAL_NEWS_SOURCES))
        self.assertEqual(
            stats["sources"],
            [source_config["name"] for source_config in HISTORICAL_NEWS_SOURCES],
        )

    def test_rss_items_keep_publication_dates_for_day_by_day_fetching(self):
        feed = b"""
        <rss><channel>
            <item>
                <title>Today story</title>
                <link>https://example.com/today</link>
                <pubDate>Thu, 11 Jun 2026 10:00:00 GMT</pubDate>
                <description>Summary</description>
            </item>
            <item>
                <title>Yesterday story</title>
                <link>https://example.com/yesterday</link>
                <pubDate>Wed, 10 Jun 2026 10:00:00 GMT</pubDate>
                <description>Summary</description>
            </item>
        </channel></rss>
        """

        items = rss_items(feed)

        self.assertEqual(items[0]["published_date"], date(2026, 6, 11))
        self.assertEqual(items[1]["published_date"], date(2026, 6, 10))


class ArticleFilterTests(TestCase):
    def test_subjectivity_filters_use_three_labels(self):
        objective = Article.objects.create(
            title="Objective article",
            topic="world",
            content="According to documents.",
            polarity=0,
            subjectivity=0.2,
            source="https://example.com/objective",
            published_date=date.today(),
        )
        insufficient = Article.objects.create(
            title="Mixed article",
            topic="world",
            content="Mixed signals.",
            polarity=0,
            subjectivity=0.4,
            source="https://example.com/mixed",
            published_date=date.today(),
        )
        subjective = Article.objects.create(
            title="Subjective article",
            topic="world",
            content="Critics say it is outrageous.",
            polarity=0,
            subjectivity=0.6,
            source="https://example.com/subjective",
            published_date=date.today(),
        )

        self.assertQuerySetEqual(
            apply_article_filters(Article.objects.all(), {"subjectivity": "objective"}),
            [objective],
        )
        self.assertQuerySetEqual(
            apply_article_filters(Article.objects.all(), {"subjectivity": "insufficient"}),
            [insufficient],
        )
        self.assertQuerySetEqual(
            apply_article_filters(Article.objects.all(), {"subjectivity": "subjective"}),
            [subjective],
        )

    def test_article_detail_renders_inline_score_badges(self):
        article = Article.objects.create(
            title="Progress confirmed",
            topic="world",
            content="Officials said progress was protected.",
            polarity=0.2,
            subjectivity=0.2,
            source="https://example.com/detail",
            published_date=date.today(),
        )

        response = Client(SERVER_NAME="localhost").get(f"/article/{article.slug}/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "P +")
        self.assertContains(response, "Subjectivité:")

    def test_word_cloud_rows_include_frequency_and_indicator_scores(self):
        Article.objects.create(
            title="Officials report progress",
            topic="world",
            content="Officials said progress was confirmed. Critics say it is terrible.",
            polarity=0,
            subjectivity=0.4,
            source="https://example.com/cloud",
            published_date=date.today(),
        )

        rows = word_cloud_rows(Article.objects.all())
        by_word = {row["word"]: row for row in rows}

        self.assertIn("progress", by_word)
        self.assertIn("officials", by_word)
        self.assertIn("critics", by_word)
        self.assertGreater(by_word["progress"]["polarity_score"], 0)
        self.assertLess(by_word["officials"]["subjectivity_score"], 0)
        self.assertGreater(by_word["critics"]["subjectivity_score"], 0)

    def test_word_cloud_visual_renders_image_and_rows(self):
        Article.objects.create(
            title="Officials report progress",
            topic="world",
            content="Officials said progress was confirmed. Critics say it is terrible.",
            polarity=0,
            subjectivity=0.4,
            source="https://example.com/cloud-plot",
            published_date=date.today(),
        )

        visual = build_word_cloud_visual(Article.objects.all())

        self.assertTrue(visual["image_url"].startswith("data:image/png;base64,"))
        self.assertIn("progress", [row["word"] for row in visual["rows"]])
