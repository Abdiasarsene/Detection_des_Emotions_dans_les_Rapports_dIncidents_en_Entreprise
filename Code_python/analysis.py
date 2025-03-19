# Importation des biblithèques
import pandas as pd
from wordcloud import WordCloud
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Imortation des données textuelles
rapport_incident = pd.read_excel(r"D:\Projects\IT\Data Science & IA\Detection_Emotions_Rapports_Incidents\Text_data\incident_analysis.xlsx")

# Nuage des mots




from transformers import pipeline

# Charger le modèle de sentiment en français
sentiment_pipeline = pipeline("sentiment-analysis", model="cmarkea/distilcamembert-base-sentiment")

# Appliquer sur tes textes
rapport_incident ["Sentiment"] = rapport_incident ["Inutilite_supr"].apply(lambda x: sentiment_pipeline(x)[0]['label'])
