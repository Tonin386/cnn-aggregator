# Generated manually for multi-source aggregation.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("cnn_aggregator", "0002_workerlog"),
    ]

    operations = [
        migrations.AddField(
            model_name="article",
            name="publisher",
            field=models.CharField(db_index=True, default="CNN", max_length=80, verbose_name="Publisher"),
        ),
        migrations.AddIndex(
            model_name="article",
            index=models.Index(fields=["publisher", "published_date"], name="cnn_aggrega_publish_5d3d1f_idx"),
        ),
    ]
