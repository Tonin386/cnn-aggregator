from datetime import date, timedelta
from unittest.mock import patch

from django.test import Client, SimpleTestCase, TestCase

from cnn_aggregator.management.commands.run_cnn_worker import Command
from cnn_aggregator.scoring_config import SCORING_VERSION
from cnn_aggregator.scoring_evaluation import SCORING_EVALUATION_CASES

from .models import Article, WorkerState
from .utils import (
    NEWS_PHRASE_ENTRIES,
    NEWS_PHRASE_LEXICON,
    OBJECTIVE_CUES,
    OBJECTIVE_PHRASE_SIGNAL_WEIGHT,
    OBJECTIVE_PHRASES,
    OBJECTIVE_SUBJECTIVITY_WEIGHT,
    SENTIMENT_LEXICON,
    SENTIMENT_WORD_CATEGORIES,
    SUBJECTIVE_CUES,
    SUBJECTIVE_PHRASE_SIGNAL_WEIGHT,
    SUBJECTIVE_PHRASES,
    SUBJECTIVE_SUBJECTIVITY_WEIGHT,
    SUBJECTIVITY_OBJECTIVE_BALANCE,
    FetchStopRequested,
    HISTORICAL_NEWS_SOURCES,
    analyze_article_sentiment,
    analyze_sentiment,
    article_scores_current,
    article_word_score_annotations,
    classify_subjectivity,
    normalize_topic,
    repeated_term_weight,
    retrieve_all_articles_for_date,
    scoring_dependency_status,
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

    def test_repeated_term_weight_caps_duplicate_content(self):
        counts = {}

        total_weight = sum(repeated_term_weight(counts, "word:record") for _ in range(100))

        self.assertAlmostEqual(total_weight, 2.5)

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

    def test_polarity_word_lexicon_is_zero_sum(self):
        raw_total = sum(
            float(entry["score"]) * len(entry["words"])
            for entry in SENTIMENT_WORD_CATEGORIES.values()
        )
        runtime_total = sum(SENTIMENT_LEXICON.values())

        self.assertAlmostEqual(raw_total, 0.0)
        self.assertAlmostEqual(runtime_total, 0.0)

    def test_polarity_phrase_lexicon_is_zero_sum(self):
        raw_total = sum(float(entry["score"]) for entry in NEWS_PHRASE_ENTRIES)
        runtime_total = sum(NEWS_PHRASE_LEXICON.values())

        self.assertAlmostEqual(raw_total, 0.0)
        self.assertAlmostEqual(runtime_total, 0.0)

    def test_combined_polarity_lexicons_are_zero_sum(self):
        raw_word_total = sum(
            float(entry["score"]) * len(entry["words"])
            for entry in SENTIMENT_WORD_CATEGORIES.values()
        )
        raw_phrase_total = sum(float(entry["score"]) for entry in NEWS_PHRASE_ENTRIES)
        runtime_total = sum(SENTIMENT_LEXICON.values()) + sum(NEWS_PHRASE_LEXICON.values())

        self.assertAlmostEqual(raw_word_total + raw_phrase_total, 0.0)
        self.assertAlmostEqual(runtime_total, 0.0)

    def test_polarity_lexicons_do_not_hide_duplicate_entries(self):
        words = [
            word
            for entry in SENTIMENT_WORD_CATEGORIES.values()
            for word in entry["words"]
        ]
        phrase_patterns = [entry["pattern"] for entry in NEWS_PHRASE_ENTRIES]

        self.assertEqual(len(words), len(set(words)))
        self.assertEqual(len(phrase_patterns), len(set(phrase_patterns)))

    def test_sports_context_dampens_competition_terms(self):
        neutral_sports_text = (
            "The World Cup group of death includes teams with a record number "
            "of wins and one defeat in qualifying."
        )
        sports_polarity, _ = analyze_sentiment(neutral_sports_text, topic="sports")
        generic_polarity, _ = analyze_sentiment(neutral_sports_text)

        self.assertLess(abs(sports_polarity), abs(generic_polarity))
        self.assertLess(abs(sports_polarity), 0.25)

    def test_opinion_topic_adds_subjectivity_prior(self):
        _, general_subjectivity = analyze_article_sentiment(
            "Congress debates Iran powers",
            "Lawmakers said the vote followed hearings and public records.",
            topic="politics",
            publisher="The Guardian",
        )
        _, opinion_subjectivity = analyze_article_sentiment(
            "Congress debates Iran powers",
            "Lawmakers said the vote followed hearings and public records.",
            topic="opinion",
            publisher="The Guardian",
        )

        self.assertGreater(opinion_subjectivity, general_subjectivity)
        self.assertGreaterEqual(opinion_subjectivity, 0.30)

    def test_fatal_incident_response_words_do_not_make_article_positive(self):
        polarity, subjectivity = analyze_article_sentiment(
            "California officials find body of missing five-year-old girl swept out to sea",
            (
                "Mother and brother were rescued after wave engulfed trio in Laguna Beach. "
                "California officials have recovered the body of a five-year-old girl who "
                "earlier this week was swept into the ocean by turbulent waters. "
                "Her disappearance set off a nearly 30-hour-long search. "
                "An aerial search located a body that was positively identified as the young girl. "
                "The mayor extended his deepest condolences to the family and called the "
                "incident heartbreaking. The city marine safety chief urged beach visitors "
                "to heed caution."
            ),
            topic="us",
            publisher="The Guardian",
        )

        self.assertLess(polarity, -0.25)
        self.assertLess(subjectivity, 0.50)

    def test_record_is_neutral_outside_performance_context(self):
        transcript_polarity, _ = analyze_sentiment(
            "The host said the interview was on the record and the recording record was updated.",
            topic="general",
        )
        business_polarity, _ = analyze_sentiment(
            "The company reported record profits and revenue growth.",
            topic="business",
        )

        self.assertLess(abs(transcript_polarity), 0.2)
        self.assertGreater(business_polarity, 0.2)

    def test_scores_expose_separate_dimensions_and_version(self):
        scores = analyze_sentiment(
            "Officials said the rescue operation was successful after a dangerous storm.",
            topic="us",
            publisher="CNN",
        )

        self.assertEqual(scores.scoring_version, SCORING_VERSION)
        self.assertIn("dependencies", scores.metadata)
        self.assertIsInstance(scores.event_polarity, float)
        self.assertIsInstance(scores.writing_polarity, float)
        self.assertEqual(scores.editorial_subjectivity, scores.subjectivity)

    def test_quoted_opinion_has_lower_editorial_subjectivity_than_direct_opinion(self):
        quoted_scores = analyze_sentiment(
            'Officials said critics called the plan "outrageous and terrible" during a hearing.',
            topic="politics",
            publisher="CNN",
        )
        direct_scores = analyze_sentiment(
            "The plan is outrageous and terrible and should be stopped.",
            topic="politics",
            publisher="CNN",
        )

        self.assertLess(quoted_scores.editorial_subjectivity, direct_scores.editorial_subjectivity)

    def test_scoring_dependency_status_is_explicit(self):
        status = scoring_dependency_status()

        self.assertEqual(status["scoring_version"], SCORING_VERSION)
        self.assertIn("textblob", status)
        self.assertIn("vader", status)

    def test_evaluation_cases_stay_within_expected_score_ranges(self):
        for case in SCORING_EVALUATION_CASES:
            with self.subTest(case=case["name"]):
                scores = analyze_sentiment(
                    case["text"],
                    topic=case["topic"],
                    publisher=case["publisher"],
                )
                expected = case["expected"]
                if "subjectivity_max" in expected:
                    self.assertLessEqual(scores.subjectivity, expected["subjectivity_max"])
                if "subjectivity_min" in expected:
                    self.assertGreaterEqual(scores.subjectivity, expected["subjectivity_min"])
                if "polarity_max" in expected:
                    self.assertLessEqual(scores.polarity, expected["polarity_max"])
                if "polarity_abs_max" in expected:
                    self.assertLessEqual(abs(scores.polarity), expected["polarity_abs_max"])
                if "editorial_subjectivity_max" in expected:
                    self.assertLessEqual(
                        scores.editorial_subjectivity,
                        expected["editorial_subjectivity_max"],
                    )
                if "writing_vs_event_abs_max" in expected:
                    self.assertLessEqual(
                        abs(scores.writing_polarity - scores.event_polarity),
                        expected["writing_vs_event_abs_max"],
                    )


class TopicRationalizationTests(SimpleTestCase):
    def test_topic_aliases_are_normalized_to_cross_source_taxonomy(self):
        self.assertEqual(normalize_topic("sport"), "sports")
        self.assertEqual(normalize_topic("football"), "sports")
        self.assertEqual(normalize_topic("commentisfree"), "opinion")
        self.assertEqual(normalize_topic("us-news"), "us")
        self.assertEqual(normalize_topic("tech"), "technology")

    def test_vague_topics_use_url_and_title_keywords(self):
        self.assertEqual(
            normalize_topic(
                "news",
                publisher="NPR",
                url="https://www.npr.org/2026/06/11/story",
                title="World Cup opener schedule and teams to watch",
            ),
            "sports",
        )
        self.assertEqual(
            normalize_topic(
                "news",
                publisher="CNN",
                url="https://www.cnn.com/2026/06/11/politics/story",
                title="Congress votes on new policy",
            ),
            "politics",
        )


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
        self.assertEqual(article.scoring_version, SCORING_VERSION)
        self.assertTrue(article.scoring_metadata)

    def test_article_scores_current_requires_current_scoring_version(self):
        article = Article.objects.create(
            title="Old score",
            topic="world",
            content="Officials said the report was released.",
            polarity=0.1,
            subjectivity=0.2,
            scoring_version="old",
            scoring_metadata={"dependencies": {}},
            source="https://example.com/old-score",
            published_date=date.today(),
        )

        self.assertFalse(article_scores_current(article))

        article.scoring_version = SCORING_VERSION
        self.assertTrue(article_scores_current(article))

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

    @patch("cnn_aggregator.views.terminate_worker_processes", return_value=1)
    def test_dashboard_stop_sets_idle_after_terminating_worker_process(self, terminate_processes):
        state = WorkerState.get_solo()
        state.status = WorkerState.STATUS_RUNNING
        state.save()

        response = Client(SERVER_NAME="localhost").post("/worker/stop/")

        self.assertEqual(response.status_code, 200)
        terminate_processes.assert_called_once()
        state.refresh_from_db()
        self.assertEqual(state.status, WorkerState.STATUS_IDLE)
        self.assertFalse(state.stop_requested)

    def test_all_source_retrieval_can_stop_between_sources(self):
        with self.assertRaises(FetchStopRequested):
            retrieve_all_articles_for_date(date.today(), stop_checker=lambda: True)

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
        historical_fetch.side_effect = lambda source_config, target_date, limit=None, log_callback=None, stop_checker=None: {
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

        cnn_fetch.assert_called_once_with(target_date, limit=5, log_callback=None, stop_checker=None)
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
