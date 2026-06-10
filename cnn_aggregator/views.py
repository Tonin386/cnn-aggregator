from django.views.decorators.csrf import csrf_exempt
from django.views.generic.detail import DetailView
from django.core.paginator import Paginator
from django.db.models import Avg, Count
from django.utils import timezone
from django.http import JsonResponse
from django.shortcuts import render
import plotly.graph_objs as go
import plotly.offline as pyo
from .utils import (
    SUBJECTIVITY_OBJECTIVE_THRESHOLD,
    SUBJECTIVITY_SUBJECTIVE_THRESHOLD,
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
