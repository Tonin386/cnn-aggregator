from django.core.management.base import BaseCommand

from cnn_aggregator.models import Article
from cnn_aggregator.utils import is_english_text, language_probabilities


class Command(BaseCommand):
    help = "Delete stored articles that are not detected as English."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be deleted without deleting articles.",
        )
        parser.add_argument(
            "--min-probability",
            type=float,
            default=0.78,
            help="Minimum English probability required to keep an article.",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        min_probability = options["min_probability"]
        total = Article.objects.count()
        removed = 0
        kept = 0

        self.stdout.write(
            f"Scanning {total} article(s) with English probability threshold {min_probability:.2f}."
        )

        for article in Article.objects.order_by("id").iterator():
            text = f"{article.title}. {article.content}"
            if is_english_text(text, min_probability=min_probability):
                kept += 1
                continue

            probabilities = language_probabilities(text)
            language_summary = ", ".join(
                f"{language.lang}:{language.prob:.2f}" for language in probabilities[:3]
            ) or "unknown"
            self.stdout.write(
                f"Deleting non-English article #{article.id} [{article.publisher}] "
                f"{language_summary}: {article.title[:120]}"
            )
            removed += 1
            if not dry_run:
                article.delete()

        action = "Would delete" if dry_run else "Deleted"
        self.stdout.write(self.style.SUCCESS(f"{action} {removed} article(s). Kept {kept}."))
