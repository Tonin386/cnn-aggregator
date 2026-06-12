from django.core.management.base import BaseCommand

from cnn_aggregator.models import Article
from cnn_aggregator.utils import analyze_article_sentiment, fix_text_encoding


class Command(BaseCommand):
    help = "Repair mojibake in stored article titles and content."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Report changes without saving.")
        parser.add_argument("--rescore", action="store_true", help="Recompute sentiment scores for repaired articles.")

    def handle(self, *args, **options):
        scanned = 0
        repaired = 0

        for article in Article.objects.iterator():
            scanned += 1
            fixed_title = fix_text_encoding(article.title)
            fixed_content = fix_text_encoding(article.content)

            if fixed_title == article.title and fixed_content == article.content:
                continue

            repaired += 1
            self.stdout.write(f"Repairing #{article.id}: {article.title} -> {fixed_title}")

            if options["dry_run"]:
                continue

            article.title = fixed_title
            article.content = fixed_content
            update_fields = ["title", "content"]

            if options["rescore"]:
                scores = analyze_article_sentiment(
                    article.title,
                    article.content,
                    topic=article.topic,
                    publisher=article.publisher,
                )
                article.polarity = scores.polarity
                article.subjectivity = scores.subjectivity
                article.event_polarity = scores.event_polarity
                article.writing_polarity = scores.writing_polarity
                article.editorial_subjectivity = scores.editorial_subjectivity
                article.scoring_version = scores.scoring_version
                article.scoring_metadata = scores.metadata
                update_fields.extend([
                    "polarity",
                    "subjectivity",
                    "event_polarity",
                    "writing_polarity",
                    "editorial_subjectivity",
                    "scoring_version",
                    "scoring_metadata",
                ])

            article.save(update_fields=update_fields)

        self.stdout.write(self.style.SUCCESS(f"Scanned {scanned} articles, repaired {repaired}."))
