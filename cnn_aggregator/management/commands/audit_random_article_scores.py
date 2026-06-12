import random
import textwrap

from django.core.management.base import BaseCommand

from cnn_aggregator.models import Article
from cnn_aggregator.utils import analyze_article_sentiment, sentiment_contributions


class Command(BaseCommand):
    help = "Print a small random sample of articles with scores and top scoring terms for manual audit."

    def add_arguments(self, parser):
        parser.add_argument("--count", type=int, default=5, help="Number of articles to sample.")
        parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible samples.")
        parser.add_argument("--topic", default="", help="Restrict the sample to one stored topic.")
        parser.add_argument("--publisher", default="", help="Restrict the sample to one publisher.")
        parser.add_argument("--contributions", type=int, default=10, help="Number of contributions to show.")
        parser.add_argument("--excerpt-chars", type=int, default=900, help="Maximum excerpt length.")

    def handle(self, *args, **options):
        queryset = Article.objects.all()
        if options["topic"]:
            queryset = queryset.filter(topic=options["topic"])
        if options["publisher"]:
            queryset = queryset.filter(publisher=options["publisher"])

        article_ids = list(queryset.values_list("id", flat=True))
        if not article_ids:
            self.stdout.write(self.style.WARNING("No articles matched the audit filters."))
            return

        rng = random.Random(options["seed"])
        sample_size = min(max(options["count"], 1), len(article_ids))
        sampled_ids = rng.sample(article_ids, sample_size)

        for index, article in enumerate(Article.objects.filter(id__in=sampled_ids), start=1):
            scores = analyze_article_sentiment(
                article.title,
                article.content,
                topic=article.topic,
                publisher=article.publisher,
            )
            score_delta = (
                abs(article.polarity - scores.polarity)
                + abs(article.subjectivity - scores.subjectivity)
            )

            self.stdout.write("\n" + "=" * 100)
            self.stdout.write(f"{index}. #{article.id} {article.title}")
            self.stdout.write(f"Source: {article.source}")
            self.stdout.write(f"Publisher/topic: {article.publisher} / {article.topic}")
            self.stdout.write(
                "Stored score: "
                f"polarity={article.polarity:.4f}, subjectivity={article.subjectivity:.4f}, "
                f"event={article.event_polarity:.4f}, writing={article.writing_polarity:.4f}, "
                f"version={article.scoring_version or '-'}"
            )
            self.stdout.write(
                "Current score: "
                f"polarity={scores.polarity:.4f}, subjectivity={scores.subjectivity:.4f}, "
                f"event={scores.event_polarity:.4f}, writing={scores.writing_polarity:.4f}, "
                f"version={scores.scoring_version}"
            )
            if score_delta > 0.001:
                self.stdout.write(self.style.WARNING(f"Score delta from stored values: {score_delta:.4f}"))

            excerpt = textwrap.shorten(
                " ".join(article.content.split()),
                width=max(options["excerpt_chars"], 120),
                placeholder="...",
            )
            self.stdout.write("\nExcerpt:")
            self.stdout.write(textwrap.fill(excerpt, width=100))

            self.stdout.write("\nTop contributions:")
            for contribution in sentiment_contributions(
                f"{article.title}. {article.content}",
                limit=max(options["contributions"], 1),
                topic=article.topic,
                publisher=article.publisher,
            ):
                self.stdout.write(
                    "  "
                    f"{contribution['term']:<32} "
                    f"{contribution['kind']:<10} "
                    f"pol={float(contribution['polarity_contribution']): .4f} "
                    f"subj={float(contribution['subjectivity_contribution']): .4f} "
                    f"hits={contribution['occurrences']}"
                )
