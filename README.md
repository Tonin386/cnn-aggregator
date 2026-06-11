# CNN News Aggregator

Authors: Antonin MATHUBERT

Project for the BHT Workflows class in Summer 2024. 
Can be run as a webapp locally or inside the Jupyter Notebook.

## Worker

The homepage reads articles already stored in the database. When the Django server starts, the worker starts automatically and fetches articles from historical news sources from today backwards: J, J-1, J-2, and so on.

Article discovery currently uses these historical indexes:

```text
https://www.cnn.com/sitemap/article.xml
https://www.aljazeera.com/sitemaps/article-new.xml
https://www.aljazeera.com/sitemaps/article-archive.xml
https://googlecrawl.npr.org/standard/sitemap_index.xml
https://googlecrawl.npr.org/news/sitemap_news.xml
https://content.guardianapis.com/search
```

The worker filters each source by the target day, fetches matching URLs, stores the publisher, and only then moves to the previous day.

The worker can be monitored from `/worker/`. For debugging, it can also be run manually with:

```bash
python manage.py run_cnn_worker
```
