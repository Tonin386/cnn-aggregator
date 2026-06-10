import os
import subprocess
import sys
from datetime import timedelta

from django.apps import AppConfig
from django.db import OperationalError, ProgrammingError
from django.utils import timezone


class CnnAggregatorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "cnn_aggregator"

    def ready(self):
        if not should_autostart_worker():
            return

        try:
            from .models import WorkerLog, WorkerState

            state = WorkerState.get_solo()
            if is_worker_recently_alive(state):
                return

            WorkerLog.write(state, "Auto-start requested by Django app startup.")
            state.status = WorkerState.STATUS_RUNNING
            state.stop_requested = False
            state.last_message = "Auto-starting worker process..."
            state.heartbeat_at = timezone.now()
            state.save()

            subprocess.Popen(
                [sys.executable, "manage.py", "run_cnn_worker"],
                cwd=os.getcwd(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OperationalError, ProgrammingError):
            return


def should_autostart_worker():
    if os.environ.get("CNN_AGGREGATOR_AUTOSTART_WORKER", "1") != "1":
        return False

    command = sys.argv[1] if len(sys.argv) > 1 else ""
    ignored_commands = {
        "check",
        "collectstatic",
        "dbshell",
        "flush",
        "makemigrations",
        "migrate",
        "run_cnn_worker",
        "shell",
        "showmigrations",
        "test",
    }
    if command in ignored_commands:
        return False

    if command == "runserver":
        return os.environ.get("RUN_MAIN") == "true" or "--noreload" in sys.argv

    return command in {"gunicorn", "uwsgi", "daphne"}


def is_worker_recently_alive(state):
    if state.status != "running" or not state.heartbeat_at:
        return False
    return timezone.now() - state.heartbeat_at < timedelta(minutes=5)
