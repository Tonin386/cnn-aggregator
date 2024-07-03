from django.db import models
from django.utils.text import slugify

class Article(models.Model):
    title = models.CharField("Title", max_length=200)
    topic = models.CharField("Topic", max_length=200)
    content = models.TextField("Content")
    polarity = models.FloatField("Polarity score")
    subjectivity = models.FloatField("Subjectivity score")
    slug = models.SlugField("Slug", unique=True, max_length=200, blank=True)
    fetch_date = models.DateField("Fetch date", auto_now_add=True)
    source = models.TextField("Source link")

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.title
