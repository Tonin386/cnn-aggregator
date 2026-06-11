import time
from datetime import date, timedelta

from django.core.management.base import BaseCommand
from django.db.models import Max
from django.utils import timezone

from cnn_aggregator.models import Article, WorkerLog, WorkerState
from cnn_aggregator.utils import (
    clear_source_discovery_caches,
    retrieve_all_articles_for_date,
    update_article_scores,
)


class Command(BaseCommand):
    help = "Fetch articles from historical news sources from today backwards, keeping progress in WorkerState."

    def add_arguments(self, parser):
        parser.add_argument("--start-date", help="Date to start from, formatted as YYYY-MM-DD.")
        parser.add_argument("--once", action="store_true", help="Process one day and exit.")
        parser.add_argument(
            "--rescore-existing",
            action="store_true",
            help="Recompute polarity and subjectivity for all stored articles before fetching.",
        )
        parser.add_argument("--sleep", type=float, default=3.0, help="Seconds to sleep between days.")
        parser.add_argument("--limit-per-day", type=int, default=None, help="Maximum URLs to fetch per day.")
        parser.add_argument(
            "--freshness-interval",
            type=float,
            default=300.0,
            help="Seconds between checks for articles newer than the latest stored article. Use 0 to disable.",
        )
        parser.add_argument(
            "--freshness-window-days",
            type=int,
            default=3,
            help="Maximum number of recent publication dates to rescan during freshness checks.",
        )

    def handle(self, *args, **options):
        state = WorkerState.get_solo()
        current_date = self._initial_date(options.get("start_date"), state)

        state.status = WorkerState.STATUS_RUNNING
        state.current_date = current_date
        state.started_at = timezone.now()
        state.heartbeat_at = timezone.now()
        state.last_message = f"Starting on {current_date.isoformat()}"
        state.last_error = ""
        state.stop_requested = False
        state.save()

        self._log(state, f"News worker started on {current_date.isoformat()}", WorkerLog.LEVEL_SUCCESS)
        if options["rescore_existing"]:
            self._rescore_existing_articles(state)
            if options["once"]:
                self._mark_idle(state, "Existing articles rescored. Worker is idle.")
                return

        last_freshness_check = timezone.now()
        self._check_fresh_articles(state, options)

        while True:
            state.refresh_from_db()
            if state.stop_requested:
                self._mark_idle(state, "Stop requested. Worker is idle.")
                return

            try:
                if self._freshness_check_due(last_freshness_check, options["freshness_interval"]):
                    self._check_fresh_articles(state, options)
                    last_freshness_check = timezone.now()
                    state.refresh_from_db()
                    if state.stop_requested:
                        self._mark_idle(state, "Stop requested. Worker is idle.")
                        return

                state.status = WorkerState.STATUS_RUNNING
                state.current_date = current_date
                state.heartbeat_at = timezone.now()
                state.last_message = f"Fetching all sources for {current_date.isoformat()}"
                state.save()
                self._log(state, state.last_message)

                stats = retrieve_all_articles_for_date(
                    current_date,
                    limit_per_source=options.get("limit_per_day"),
                    log_callback=lambda message, level=WorkerLog.LEVEL_INFO: self._log(
                        state,
                        message,
                        level,
                    ),
                )

                state.refresh_from_db()
                state.total_seen += stats["seen"]
                state.total_created += stats["created"]
                state.total_updated += stats["updated"]
                state.heartbeat_at = timezone.now()
                state.last_message = (
                    f"{current_date.isoformat()}: "
                    f"{stats['discovered']} discovered from all sources, "
                    f"{stats['created']} created, {stats['updated']} updated, "
                    f"{stats['skipped']} skipped, {stats['errors']} errors."
                )
                state.last_error = "" if stats["errors"] == 0 else state.last_error
                state.current_date = current_date - timedelta(days=1)
                state.save()

                level = WorkerLog.LEVEL_WARNING if stats["errors"] else WorkerLog.LEVEL_SUCCESS
                self._log(state, state.last_message, level)
            except Exception as exc:
                state.refresh_from_db()
                state.status = WorkerState.STATUS_ERROR
                state.heartbeat_at = timezone.now()
                state.last_error = str(exc)
                state.last_message = f"Worker failed on {current_date.isoformat()}"
                state.save()
                self._log(state, f"{state.last_message}: {exc}", WorkerLog.LEVEL_ERROR)
                raise

            if options.get("once"):
                self._mark_idle(state, "One-day worker run completed.")
                return

            current_date -= timedelta(days=1)
            time.sleep(options["sleep"])

    def _initial_date(self, start_date, state):
        if start_date:
            return date.fromisoformat(start_date)
        return state.current_date or date.today()

    def _mark_idle(self, state, message):
        state.refresh_from_db()
        state.status = WorkerState.STATUS_IDLE
        state.heartbeat_at = timezone.now()
        state.last_message = message
        state.stop_requested = False
        state.save()
        self._log(state, message, WorkerLog.LEVEL_SUCCESS)

    def _log(self, state, message, level=WorkerLog.LEVEL_INFO):
        WorkerLog.write(state, message, level)
        if level == WorkerLog.LEVEL_ERROR:
            self.stderr.write(self.style.ERROR(message))
        elif level in {WorkerLog.LEVEL_SUCCESS, WorkerLog.LEVEL_WARNING}:
            self.stdout.write(self.style.SUCCESS(message))
        else:
            self.stdout.write(message)

    def _freshness_check_due(self, last_freshness_check, interval_seconds):
        if not interval_seconds or interval_seconds <= 0:
            return False
        return (timezone.now() - last_freshness_check).total_seconds() >= interval_seconds

    def _latest_article_date(self):
        return (
            Article.objects
            .exclude(published_date__isnull=True)
            .aggregate(value=Max("published_date"))["value"]
        )

    def _freshness_dates(self, window_days):
        today = date.today()
        latest_date = self._latest_article_date() or today
        start_date = min(latest_date, today)
        max_days = max(window_days or 1, 1)
        dates = []

        cursor = today
        while cursor >= start_date and len(dates) < max_days:
            dates.append(cursor)
            cursor -= timedelta(days=1)

        return dates, latest_date

    def _clear_sitemap_caches(self):
        clear_source_discovery_caches()

    def _rescore_existing_articles(self, state):
        total = Article.objects.count()
        self._log(state, f"Rescoring {total} stored articles.", WorkerLog.LEVEL_WARNING)

        for article_index, article in enumerate(Article.objects.order_by("id").iterator(), start=1):
            state.refresh_from_db()
            if state.stop_requested:
                self._log(state, "Rescore stopped by request.", WorkerLog.LEVEL_WARNING)
                return

            old_polarity = article.polarity
            old_subjectivity = article.subjectivity
            update_article_scores(article, force=True)

            if article_index == 1 or article_index % 25 == 0 or article_index == total:
                state.heartbeat_at = timezone.now()
                state.last_message = f"Rescored {article_index}/{total} stored articles."
                state.save()
                self._log(
                    state,
                    (
                        f"Rescored {article_index}/{total}: {article.title[:90]} "
                        f"(polarity {old_polarity:.4f}->{article.polarity:.4f}, "
                        f"subjectivity {old_subjectivity:.4f}->{article.subjectivity:.4f})"
                    ),
                )

        self._log(state, f"Finished rescoring {total} stored articles.", WorkerLog.LEVEL_SUCCESS)

    def _check_fresh_articles(self, state, options):
        dates_to_scan, latest_date = self._freshness_dates(options["freshness_window_days"])
        self._clear_sitemap_caches()
        self._log(
            state,
            (
                "Freshness check started: "
                f"latest stored article date is {latest_date.isoformat()}, "
                f"rescanning {len(dates_to_scan)} recent day(s)."
            ),
        )

        for target_date in dates_to_scan:
            state.refresh_from_db()
            if state.stop_requested:
                return

            state.status = WorkerState.STATUS_RUNNING
            state.current_date = target_date
            state.heartbeat_at = timezone.now()
            state.last_message = f"Checking for fresh articles on {target_date.isoformat()}"
            state.save()
            self._log(state, state.last_message)

            stats = retrieve_all_articles_for_date(
                target_date,
                limit_per_source=options.get("limit_per_day"),
                log_callback=lambda message, level=WorkerLog.LEVEL_INFO: self._log(
                    state,
                    message,
                    level,
                ),
            )

            state.refresh_from_db()
            state.total_seen += stats["seen"]
            state.total_created += stats["created"]
            state.total_updated += stats["updated"]
            state.heartbeat_at = timezone.now()
            state.last_message = (
                f"Freshness {target_date.isoformat()}: "
                f"{stats['discovered']} discovered, {stats['created']} new, "
                f"{stats['updated']} updated, {stats['skipped']} skipped, "
                f"{stats['errors']} errors."
            )
            if stats["errors"] == 0:
                state.last_error = ""
            state.save()

            level = WorkerLog.LEVEL_WARNING if stats["errors"] else WorkerLog.LEVEL_SUCCESS
            self._log(state, state.last_message, level)
