from django.db import models
from django.utils.text import slugify

class Article(models.Model):
    title = models.CharField("Title", max_length=200)
    publisher = models.CharField("Publisher", max_length=80, default="CNN", db_index=True)
    topic = models.CharField("Topic", max_length=200)
    content = models.TextField("Content")
    polarity = models.FloatField("Polarity score")
    subjectivity = models.FloatField("Subjectivity score")
    event_polarity = models.FloatField("Event polarity score", default=0.0)
    writing_polarity = models.FloatField("Writing polarity score", default=0.0)
    editorial_subjectivity = models.FloatField("Editorial subjectivity score", default=0.0)
    scoring_version = models.CharField("Scoring version", max_length=32, blank=True, default="")
    scoring_metadata = models.JSONField("Scoring metadata", default=dict, blank=True)
    slug = models.SlugField("Slug", unique=True, max_length=200, blank=True)
    fetch_date = models.DateField("Fetch date", auto_now_add=True)
    fetched_at = models.DateTimeField("Fetched at", auto_now_add=True)
    updated_at = models.DateTimeField("Updated at", auto_now=True)
    published_date = models.DateField("Published date", null=True, blank=True, db_index=True)
    source = models.TextField("Source link")

    def save(self, *args, **kwargs):
        if not self.slug:
            base_slug = slugify(self.title)[:180] or "article"
            slug = base_slug
            suffix = 2
            while Article.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                slug = f"{base_slug}-{suffix}"
                suffix += 1
            self.slug = slug
        super().save(*args, **kwargs)

    def __str__(self):
        return self.title

    class Meta:
        ordering = ["-published_date", "-fetch_date", "-id"]
        indexes = [
            models.Index(fields=["topic", "published_date"]),
            models.Index(fields=["publisher", "published_date"]),
            models.Index(fields=["polarity"]),
            models.Index(fields=["subjectivity"]),
            models.Index(fields=["event_polarity"]),
            models.Index(fields=["writing_polarity"]),
            models.Index(fields=["editorial_subjectivity"]),
            models.Index(fields=["scoring_version"]),
        ]


class WorkerState(models.Model):
    STATUS_IDLE = "idle"
    STATUS_RUNNING = "running"
    STATUS_STOPPING = "stopping"
    STATUS_ERROR = "error"

    STATUS_CHOICES = [
        (STATUS_IDLE, "Idle"),
        (STATUS_RUNNING, "Running"),
        (STATUS_STOPPING, "Stopping"),
        (STATUS_ERROR, "Error"),
    ]

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_IDLE)
    current_date = models.DateField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    heartbeat_at = models.DateTimeField(null=True, blank=True)
    last_message = models.CharField(max_length=300, blank=True)
    last_error = models.TextField(blank=True)
    total_seen = models.PositiveIntegerField(default=0)
    total_created = models.PositiveIntegerField(default=0)
    total_updated = models.PositiveIntegerField(default=0)
    stop_requested = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

    @classmethod
    def get_solo(cls):
        state, _ = cls.objects.get_or_create(pk=1)
        return state

    def __str__(self):
        return f"News worker: {self.status}"


class WorkerLog(models.Model):
    LEVEL_INFO = "info"
    LEVEL_SUCCESS = "success"
    LEVEL_WARNING = "warning"
    LEVEL_ERROR = "error"

    LEVEL_CHOICES = [
        (LEVEL_INFO, "Info"),
        (LEVEL_SUCCESS, "Success"),
        (LEVEL_WARNING, "Warning"),
        (LEVEL_ERROR, "Error"),
    ]

    state = models.ForeignKey(WorkerState, on_delete=models.CASCADE, related_name="logs")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    level = models.CharField(max_length=20, choices=LEVEL_CHOICES, default=LEVEL_INFO)
    message = models.CharField(max_length=500)

    class Meta:
        ordering = ["-created_at", "-id"]

    @classmethod
    def write(cls, state, message, level=LEVEL_INFO):
        log = cls.objects.create(state=state, message=message[:500], level=level)
        old_logs = cls.objects.filter(state=state).order_by("-created_at", "-id")[1000:]
        old_log_ids = [old_log.id for old_log in old_logs]
        if old_log_ids:
            cls.objects.filter(id__in=old_log_ids).delete()
        return log

    def __str__(self):
        return self.message
