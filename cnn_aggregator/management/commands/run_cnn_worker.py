import time
from datetime import date, timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from cnn_aggregator.models import WorkerLog, WorkerState
from cnn_aggregator.utils import retrieve_cnn_articles_for_date


class Command(BaseCommand):
    help = "Fetch CNN articles from today backwards, keeping progress in WorkerState."

    def add_arguments(self, parser):
        parser.add_argument("--start-date", help="Date to start from, formatted as YYYY-MM-DD.")
        parser.add_argument("--once", action="store_true", help="Process one day and exit.")
        parser.add_argument("--sleep", type=float, default=3.0, help="Seconds to sleep between days.")
        parser.add_argument("--limit-per-day", type=int, default=None, help="Maximum URLs to fetch per day.")

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

        self._log(state, f"CNN worker started on {current_date.isoformat()}", WorkerLog.LEVEL_SUCCESS)

        while True:
            state.refresh_from_db()
            if state.stop_requested:
                self._mark_idle(state, "Stop requested. Worker is idle.")
                return

            try:
                state.status = WorkerState.STATUS_RUNNING
                state.current_date = current_date
                state.heartbeat_at = timezone.now()
                state.last_message = f"Fetching articles for {current_date.isoformat()}"
                state.save()
                self._log(state, state.last_message)

                stats = retrieve_cnn_articles_for_date(
                    current_date,
                    limit=options.get("limit_per_day"),
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
                    f"{stats['discovered']} discovered from {stats['sitemaps']} sitemaps, "
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
