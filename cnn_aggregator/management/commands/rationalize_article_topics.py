from collections import Counter

from django.core.management.base import BaseCommand

from cnn_aggregator.models import Article
from cnn_aggregator.utils import normalize_topic


class Command(BaseCommand):
    help = "Normalize stored article topics to the shared cross-source taxonomy."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show changes without updating articles.",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        total = Article.objects.count()
        changed = 0
        transitions = Counter()

        self.stdout.write(f"Scanning {total} article(s) for topic rationalization.")

        for article in Article.objects.order_by("id").iterator():
            normalized_topic = normalize_topic(
                article.topic,
                publisher=article.publisher,
                url=article.source,
                title=article.title,
            )
            if normalized_topic == article.topic:
                continue

            transitions[(article.topic, normalized_topic)] += 1
            changed += 1
            if not dry_run:
                article.topic = normalized_topic
                article.save(update_fields=["topic"])

        for (old_topic, new_topic), count in transitions.most_common(40):
            self.stdout.write(f"{old_topic} -> {new_topic}: {count}")

        action = "Would update" if dry_run else "Updated"
        self.stdout.write(self.style.SUCCESS(f"{action} {changed} article(s)."))
