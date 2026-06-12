from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("cnn_aggregator", "0003_article_publisher"),
    ]

    operations = [
        migrations.AddField(
            model_name="article",
            name="editorial_subjectivity",
            field=models.FloatField(default=0.0, verbose_name="Editorial subjectivity score"),
        ),
        migrations.AddField(
            model_name="article",
            name="event_polarity",
            field=models.FloatField(default=0.0, verbose_name="Event polarity score"),
        ),
        migrations.AddField(
            model_name="article",
            name="scoring_metadata",
            field=models.JSONField(blank=True, default=dict, verbose_name="Scoring metadata"),
        ),
        migrations.AddField(
            model_name="article",
            name="scoring_version",
            field=models.CharField(blank=True, default="", max_length=32, verbose_name="Scoring version"),
        ),
        migrations.AddField(
            model_name="article",
            name="writing_polarity",
            field=models.FloatField(default=0.0, verbose_name="Writing polarity score"),
        ),
        migrations.AddIndex(
            model_name="article",
            index=models.Index(fields=["event_polarity"], name="cnn_aggrega_event_p_9718a8_idx"),
        ),
        migrations.AddIndex(
            model_name="article",
            index=models.Index(fields=["writing_polarity"], name="cnn_aggrega_writing_9f5f18_idx"),
        ),
        migrations.AddIndex(
            model_name="article",
            index=models.Index(fields=["editorial_subjectivity"], name="cnn_aggrega_editori_0a7d6b_idx"),
        ),
        migrations.AddIndex(
            model_name="article",
            index=models.Index(fields=["scoring_version"], name="cnn_aggrega_scoring_3e28b7_idx"),
        ),
    ]
