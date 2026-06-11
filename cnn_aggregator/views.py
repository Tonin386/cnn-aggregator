from django.views.decorators.csrf import csrf_exempt
from django.views.generic.detail import DetailView
from django.core.paginator import Paginator
from django.core.cache import cache
from django.db.models import Avg, Count, Max
from django.utils import timezone
from django.http import JsonResponse
from django.shortcuts import render
import plotly.graph_objs as go
import plotly.offline as pyo
from io import BytesIO
import base64
import hashlib
from wordcloud import WordCloud
from .utils import (
    OBJECTIVE_CUES,
    SENTIMENT_LEXICON,
    SUBJECTIVITY_OBJECTIVE_THRESHOLD,
    SUBJECTIVITY_SUBJECTIVE_THRESHOLD,
    SUBJECTIVE_CUES,
    WORD_PATTERN,
    article_highlight_word_groups,
    sentiment_contributions,
)
from .models import Article, WorkerLog, WorkerState
from datetime import timedelta
import os
import subprocess
import sys

def home(request):
    articles_queryset = apply_article_filters(Article.objects.all(), request.GET)
    topic_options = Article.objects.order_by("topic").values_list("topic", flat=True).distinct()
    paginator = Paginator(articles_queryset, 16)
    page_obj = paginator.get_page(request.GET.get("page"))
    articles = page_obj.object_list
    worker_state = WorkerState.get_solo()

    topic_graph = build_topic_graph(articles_queryset)
    sentiment_graph = build_sentiment_graph(articles_queryset)
    timeline_graph = build_timeline_graph(articles_queryset)
    subjectivity_timeline_graph = build_subjectivity_timeline_graph(articles_queryset)
    topic_polarity_graph = build_topic_polarity_graph(articles_queryset)
    subjectivity_graph = build_subjectivity_graph(articles_queryset)
    word_cloud_visual = build_word_cloud_visual(articles_queryset)

    total_articles = articles_queryset.count()
    total_topics = articles_queryset.values("topic").distinct().count()
    avg_polarity = articles_queryset.aggregate(value=Avg("polarity"))["value"] or 0
    avg_subjectivity = articles_queryset.aggregate(value=Avg("subjectivity"))["value"] or 0
    global_article_count = Article.objects.count()
    refresh_snapshot = {
        "filtered_articles": total_articles,
        "global_articles": global_article_count,
        "topics": total_topics,
        "avg_polarity": float(avg_polarity),
        "avg_subjectivity": float(avg_subjectivity),
        "worker_seen": worker_state.total_seen,
        "worker_created": worker_state.total_created,
        "worker_updated": worker_state.total_updated,
        "worker_status": worker_state.status,
        "captured_at": timezone.now().isoformat(),
    }
    filter_params = request.GET.copy()
    filter_params.pop("page", None)
    filter_querystring = filter_params.urlencode()
    bounds = [-1, -.33, .33, 1]
    subjectivity_objective_threshold = SUBJECTIVITY_OBJECTIVE_THRESHOLD
    subjectivity_subjective_threshold = SUBJECTIVITY_SUBJECTIVE_THRESHOLD

    return render(request, "home.html", locals())

def apply_article_filters(queryset, params):
    query = params.get("q", "").strip()
    topic = params.get("topic", "").strip()
    date_from = params.get("date_from", "").strip()
    date_to = params.get("date_to", "").strip()
    sentiment = params.get("sentiment", "").strip()
    subjectivity = params.get("subjectivity", "").strip()

    if query:
        queryset = queryset.filter(title__icontains=query)
    if topic:
        queryset = queryset.filter(topic=topic)
    if date_from:
        queryset = queryset.filter(published_date__gte=date_from)
    if date_to:
        queryset = queryset.filter(published_date__lte=date_to)
    if sentiment == "negative":
        queryset = queryset.filter(polarity__lt=-0.33)
    elif sentiment == "neutral":
        queryset = queryset.filter(polarity__gte=-0.33, polarity__lte=0.33)
    elif sentiment == "positive":
        queryset = queryset.filter(polarity__gt=0.33)
    if subjectivity == "objective":
        queryset = queryset.filter(subjectivity__lt=SUBJECTIVITY_OBJECTIVE_THRESHOLD)
    elif subjectivity == "insufficient":
        queryset = queryset.filter(
            subjectivity__gte=SUBJECTIVITY_OBJECTIVE_THRESHOLD,
            subjectivity__lt=SUBJECTIVITY_SUBJECTIVE_THRESHOLD,
        )
    elif subjectivity == "subjective":
        queryset = queryset.filter(subjectivity__gte=SUBJECTIVITY_SUBJECTIVE_THRESHOLD)

    return queryset

def plot(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=55, b=45),
        height=360,
        font=dict(color="#f8f9fa"),
    )
    return pyo.plot(
        fig,
        output_type="div",
        include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True},
    )

def build_topic_graph(queryset):
    rows = list(queryset.values("topic").annotate(total=Count("id")).order_by("-total")[:12])
    fig = go.Figure(data=[go.Bar(
        x=[row["topic"].title() for row in rows],
        y=[row["total"] for row in rows],
        marker_color="#4dabf7",
        text=[row["total"] for row in rows],
        textposition="outside",
    )])
    fig.update_layout(title="Topics les plus représentés", xaxis_title="", yaxis_title="Articles")
    return plot(fig)

def build_sentiment_graph(queryset):
    negative = queryset.filter(polarity__lt=-0.33).count()
    neutral = queryset.filter(polarity__gte=-0.33, polarity__lte=0.33).count()
    positive = queryset.filter(polarity__gt=0.33).count()
    fig = go.Figure(data=[go.Pie(
        labels=["Négatif", "Neutre", "Positif"],
        values=[negative, neutral, positive],
        hole=.48,
        marker=dict(colors=["#fa5252", "#dee2e6", "#51cf66"]),
    )])
    fig.update_layout(title="Répartition du ton")
    return plot(fig)

def build_timeline_graph(queryset):
    rows = list(
        queryset
        .exclude(published_date__isnull=True)
        .values("published_date")
        .annotate(total=Count("id"), polarity=Avg("polarity"))
        .order_by("published_date")
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[row["published_date"] for row in rows],
        y=[row["total"] for row in rows],
        name="Articles",
        marker_color="#4dabf7",
        opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=[row["published_date"] for row in rows],
        y=[row["polarity"] for row in rows],
        mode="lines+markers",
        name="Polarité moyenne",
        yaxis="y2",
        line=dict(color="#ff922b"),
    ))
    fig.update_layout(
        title="Volume et polarité dans le temps",
        yaxis=dict(title="Articles"),
        yaxis2=dict(title="Polarité", overlaying="y", side="right", range=[-1, 1]),
        legend=dict(orientation="h"),
        bargap=0.25,
    )
    return plot(fig)

def build_subjectivity_graph(queryset):
    rows = list(queryset.values("topic").annotate(value=Avg("subjectivity"), total=Count("id")).order_by("-total")[:12])
    fig = go.Figure(data=[go.Bar(
        x=[row["topic"].title() for row in rows],
        y=[row["value"] for row in rows],
        marker_color="#9775fa",
        text=["%.2f" % (row["value"] or 0) for row in rows],
        textposition="outside",
    )])
    fig.update_layout(title="Subjectivité moyenne par topic", xaxis_title="", yaxis_title="Score")
    return plot(fig)

def build_topic_polarity_graph(queryset):
    rows = list(queryset.values("topic").annotate(value=Avg("polarity"), total=Count("id")).order_by("-total")[:12])
    fig = go.Figure(data=[go.Bar(
        x=[row["topic"].title() for row in rows],
        y=[row["value"] for row in rows],
        marker_color=[
            "#51cf66" if (row["value"] or 0) > 0.33 else "#fa5252" if (row["value"] or 0) < -0.33 else "#dee2e6"
            for row in rows
        ],
        text=["%.2f" % (row["value"] or 0) for row in rows],
        textposition="outside",
    )])
    fig.update_layout(
        title="Polarité moyenne par topic",
        xaxis_title="",
        yaxis_title="Score",
        yaxis=dict(range=[-1, 1]),
    )
    return plot(fig)

def build_subjectivity_timeline_graph(queryset):
    rows = list(
        queryset
        .exclude(published_date__isnull=True)
        .values("published_date")
        .annotate(total=Count("id"), subjectivity=Avg("subjectivity"))
        .order_by("published_date")
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[row["published_date"] for row in rows],
        y=[row["total"] for row in rows],
        name="Articles",
        marker_color="#4dabf7",
        opacity=0.55,
    ))
    fig.add_trace(go.Scatter(
        x=[row["published_date"] for row in rows],
        y=[row["subjectivity"] for row in rows],
        mode="lines+markers",
        name="Subjectivité moyenne",
        yaxis="y2",
        line=dict(color="#b197fc"),
    ))
    fig.update_layout(
        title="Volume et subjectivité dans le temps",
        yaxis=dict(title="Articles"),
        yaxis2=dict(title="Subjectivité", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h"),
        bargap=0.25,
    )
    return plot(fig)

WORD_CLOUD_ARTICLE_LIMIT = 350
WORD_CLOUD_CACHE_SECONDS = 300

def word_cloud_rows(queryset, limit=70, article_limit=WORD_CLOUD_ARTICLE_LIMIT):
    words = {}
    article_rows = queryset.values_list("title", "content")[:article_limit]

    for title, content in article_rows:
        tokens = WORD_PATTERN.findall(f"{title or ''} {content or ''}".lower())
        seen_in_article = set()
        for token in tokens:
            polarity_score = SENTIMENT_LEXICON.get(token, 0)
            subjectivity_score = 0
            if token in OBJECTIVE_CUES:
                subjectivity_score -= 0.35
            if token in SUBJECTIVE_CUES:
                subjectivity_score += 0.45

            if polarity_score == 0 and subjectivity_score == 0:
                continue

            row = words.setdefault(
                token,
                {
                    "word": token,
                    "count": 0,
                    "article_count": 0,
                    "polarity_score": polarity_score,
                    "subjectivity_score": subjectivity_score,
                },
            )
            row["count"] += 1
            if token not in seen_in_article:
                row["article_count"] += 1
                seen_in_article.add(token)

    for row in words.values():
        polarity_impact = abs(row["polarity_score"]) * row["count"]
        subjectivity_impact = abs(row["subjectivity_score"]) * row["count"]
        row["impact"] = polarity_impact + subjectivity_impact
        row["dominant_impact"] = max(polarity_impact, subjectivity_impact)
        row["category"] = word_cloud_category(row, polarity_impact, subjectivity_impact)

    return sorted(
        words.values(),
        key=lambda row: (row["impact"], row["article_count"], row["count"]),
        reverse=True,
    )[:limit]

def word_cloud_category(row, polarity_impact, subjectivity_impact):
    if subjectivity_impact > polarity_impact:
        return "Objectivité" if row["subjectivity_score"] < 0 else "Subjectivité"
    if row["polarity_score"] > 0:
        return "Polarité positive"
    if row["polarity_score"] < 0:
        return "Polarité négative"
    return "Mixte"

def word_cloud_color(category):
    return {
        "Polarité positive": "#69db7c",
        "Polarité négative": "#ff8787",
        "Objectivité": "#7dd3fc",
        "Subjectivité": "#c084fc",
    }.get(category, "#ffd43b")

def word_cloud_cache_key(queryset):
    aggregate = queryset.aggregate(total=Count("id"), latest_update=Max("updated_at"))
    raw_key = "|".join([
        str(queryset.query),
        str(aggregate["total"] or 0),
        str(aggregate["latest_update"] or ""),
        str(WORD_CLOUD_ARTICLE_LIMIT),
    ])
    return f"word-cloud:{hashlib.md5(raw_key.encode('utf-8')).hexdigest()}"

def build_word_cloud_visual(queryset):
    cache_key = word_cloud_cache_key(queryset)
    cached_visual = cache.get(cache_key)
    if cached_visual is not None:
        return cached_visual

    rows = word_cloud_rows(queryset)
    if not rows:
        empty_visual = {"image_url": "", "rows": [], "article_limit": WORD_CLOUD_ARTICLE_LIMIT}
        cache.set(cache_key, empty_visual, WORD_CLOUD_CACHE_SECONDS)
        return empty_visual

    categories_by_word = {
        row["word"]: row["category"]
        for row in rows
    }
    frequencies = {
        row["word"]: float(row["impact"])
        for row in rows
        if row["impact"] > 0
    }

    cloud = WordCloud(
        width=1100,
        height=380,
        background_color=None,
        mode="RGBA",
        prefer_horizontal=0.92,
        collocations=False,
        max_words=60,
        min_font_size=12,
        max_font_size=76,
        relative_scaling=0.55,
        random_state=42,
        margin=2,
    ).generate_from_frequencies(frequencies)
    cloud.recolor(
        color_func=lambda word, *args, **kwargs: word_cloud_color(categories_by_word.get(word, "Mixte")),
        random_state=42,
    )

    image_buffer = BytesIO()
    cloud.to_image().save(image_buffer, format="PNG")
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode("ascii")

    visual = {
        "image_url": f"data:image/png;base64,{image_base64}",
        "rows": rows[:18],
        "article_limit": WORD_CLOUD_ARTICLE_LIMIT,
    }
    cache.set(cache_key, visual, WORD_CLOUD_CACHE_SECONDS)
    return visual

def worker_dashboard(request):
    state = WorkerState.get_solo()
    return render(request, "worker_dashboard.html", {"state": state})

def worker_status(request):
    state = WorkerState.get_solo()
    return JsonResponse(serialize_worker_state(state))

@csrf_exempt
def start_worker(request):
    state = WorkerState.get_solo()
    if is_worker_recently_alive(state):
        return JsonResponse({"status": "already_running", "worker": serialize_worker_state(state)})

    WorkerLog.write(state, "Start requested from dashboard.")
    launch_worker_process(state, ["run_cnn_worker"], "Starting worker process...")
    return JsonResponse({"status": "started", "worker": serialize_worker_state(state)})

@csrf_exempt
def rescore_worker(request):
    state = WorkerState.get_solo()
    if is_worker_recently_alive(state):
        return JsonResponse({"status": "already_running", "worker": serialize_worker_state(state)})

    continue_after = request.GET.get("continue") == "1"
    command_args = ["run_cnn_worker", "--rescore-existing"]
    if not continue_after:
        command_args.append("--once")

    if continue_after:
        message = "Rescore requested from dashboard, then continue worker."
        launch_message = "Starting rescore, then worker process..."
    else:
        message = "Rescore-only requested from dashboard."
        launch_message = "Starting rescore-only process..."

    WorkerLog.write(state, message, WorkerLog.LEVEL_WARNING)
    launch_worker_process(state, command_args, launch_message)
    return JsonResponse({"status": "started", "worker": serialize_worker_state(state)})

@csrf_exempt
def stop_worker(request):
    state = WorkerState.get_solo()
    WorkerLog.write(state, "Stop requested from dashboard.", WorkerLog.LEVEL_WARNING)
    state.stop_requested = True
    state.status = WorkerState.STATUS_STOPPING
    state.last_message = "Stop requested from dashboard."
    state.save()
    return JsonResponse({"status": "stopping", "worker": serialize_worker_state(state)})

def launch_worker_process(state, command_args, launch_message):
    state.status = WorkerState.STATUS_RUNNING
    state.stop_requested = False
    state.last_message = launch_message
    state.heartbeat_at = timezone.now()
    state.save()

    subprocess.Popen(
        [sys.executable, "manage.py", *command_args],
        cwd=os.getcwd(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

def is_worker_recently_alive(state):
    if state.status != WorkerState.STATUS_RUNNING or not state.heartbeat_at:
        return False
    return timezone.now() - state.heartbeat_at < timedelta(minutes=5)

def serialize_worker_state(state):
    return {
        "status": state.status,
        "current_date": state.current_date.isoformat() if state.current_date else None,
        "started_at": state.started_at.isoformat() if state.started_at else None,
        "heartbeat_at": state.heartbeat_at.isoformat() if state.heartbeat_at else None,
        "last_message": state.last_message,
        "last_error": state.last_error,
        "total_seen": state.total_seen,
        "total_created": state.total_created,
        "total_updated": state.total_updated,
        "stop_requested": state.stop_requested,
        "logs": [
            {
                "created_at": log.created_at.isoformat(),
                "level": log.level,
                "message": log.message,
            }
            for log in state.logs.order_by("-created_at", "-id")[:150]
        ],
    }

class ArticleDetailView(DetailView):
    model = Article
    template_name = 'article_detail.html'
    context_object_name = 'article'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(article_highlight_word_groups())
        context["words"] = self.object.content.split(" ")
        context["sentiment_contributions"] = sentiment_contributions(
            f"{self.object.title}. {self.object.content}"
        )
        context["subjectivity_objective_threshold"] = SUBJECTIVITY_OBJECTIVE_THRESHOLD
        context["subjectivity_subjective_threshold"] = SUBJECTIVITY_SUBJECTIVE_THRESHOLD

        return context
