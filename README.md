# CNN News Aggregator

Authors: Antonin MATHUBERT

Project for the BHT Workflows class in Summer 2024. 
Can be run as a webapp locally or inside the Jupyter Notebook.

## Worker

The homepage reads articles already stored in the database. When the Django server starts, the worker starts automatically and fetches CNN articles from today backwards: J, J-1, J-2, and so on.

Article discovery uses CNN's public sitemap index:

```text
https://www.cnn.com/sitemap/article.xml
```

The worker filters that index by month, reads the matching section sitemaps, then fetches article URLs matching the target day.

The worker can be monitored from `/worker/`. For debugging, it can also be run manually with:

```bash
python manage.py run_cnn_worker
```
