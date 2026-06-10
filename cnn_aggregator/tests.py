from django.test import SimpleTestCase

from .utils import analyze_sentiment, repeated_term_weight


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
