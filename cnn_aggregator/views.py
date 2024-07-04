from django.views.decorators.csrf import csrf_exempt
from django.views.generic.detail import DetailView
from .utils import retrieve_cnn_homepage
from django.http import JsonResponse
from django.shortcuts import render
from collections import Counter
import plotly.graph_objs as go
import plotly.offline as pyo
from .models import Article
from datetime import date
import numpy as np

def home(request):
    articles = Article.objects.filter(fetch_date=date.today())
    topics_values = articles.values_list('topic', flat=True)

    # Count occurrences of each topic
    topics = Counter(topics_values)
    sorted_topics = {k: v for k, v in sorted(topics.items(), key=lambda item: item[1], reverse=True)}

    words = list(sorted_topics.keys())
    counts = list(sorted_topics.values())
    fig = go.Figure(data=[go.Bar(x=words, y=counts, text=counts, textposition='outside')])

    fig.update_layout(
        title="Today's topics on CNN front page",
        xaxis_title='Topics',
        yaxis_title='Occurrences',
        template='plotly_dark'
    )
    todays_topic_graph = pyo.plot(fig, output_type='div', include_plotlyjs=True)

    topic_polarity_means = []
    for word in words:
        topic_polarity_means.append(np.mean(articles.filter(topic=word).values_list('polarity', flat=True)))

    fig = go.Figure(data=[go.Bar(x=words, y=topic_polarity_means, text=["%.3f" % value for value in topic_polarity_means], textposition='outside')])

    fig.update_layout(
        title="Average sentiment polarity score per topic",
        xaxis_title='Topics',
        yaxis_title='Mean value',
        template='plotly_dark'
    )
    todays_topic_polarity_graph = pyo.plot(fig, output_type='div', include_plotlyjs=True)


    topic_subjectivity_means = []
    for word in words:
        topic_subjectivity_means.append(np.mean(articles.filter(topic=word).values_list('subjectivity', flat=True)))

    fig = go.Figure(data=[go.Bar(x=words, y=topic_subjectivity_means, text=["%.3f" % value for value in topic_subjectivity_means], textposition='outside')])

    fig.update_layout(
        title="Average sentiment subjectivity score per topic",
        xaxis_title='Topics',
        yaxis_title='Mean value',
        template='plotly_dark'
    )
    todays_topic_subjectivity_graph = pyo.plot(fig, output_type='div', include_plotlyjs=True)

    bounds = [-1, -.33, .33, 1]

    return render(request, "home.html", locals())

@csrf_exempt
def sync_cnn(request):
    print("Updating database...")
    retrieve_cnn_homepage()
    print("Done!")
    return JsonResponse({"status": "success"})

class ArticleDetailView(DetailView):
    model = Article
    template_name = 'article_detail.html'
    context_object_name = 'article'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        context["disturbing"] = [
            "apocalypse", "disaster", "crisis", "catastrophe", "threat",
            "terror", "violence", "fear", "destruction", "chaos", "danger",
            "terrible", "atrocious", "calamity", "tragedy", "carnage", 'murder',
            "war", "menace", "critical", "attack", "attacks", "devastating"
        ]

        context["negative"] = [
            "despair", "suffering", "misery", "anguish", "pain",
            "grief", "sadness", "melancholy", "depression", "anxiety",
            "issue", "problem", "bad", "vulnerable", "sad", "concern",
            "concerning", "worried", "dead", "death"
        ]

        context["neutral"] = [
            "routine", "commonplace", "ordinary", "standard", "typical",
            "usual", "unremarkable", "mundane", "average", "common", "boring"
        ]

        context["positive"] = [
            "hope", "comfort", "relief", "contentment", "satisfaction",
            "joy", "happiness", "love", "excitement", "enthusiasm", "good",
            "agreement", "effort", "happy", "hopeful", "proud", "life", "hope", 
            "hopeful", "support", "efforts"
        ]

        context["optimistic"] = [
            "potential", "prosperity", "success", "wonderful", "perfect", 
            "achievement", "growth", "fulfillment", "bliss", "euphoria", "peace",
            "extasis", "successful", "greatness", "incredible"
        ]

        context["words"] = self.object.content.split(" ")

        return context