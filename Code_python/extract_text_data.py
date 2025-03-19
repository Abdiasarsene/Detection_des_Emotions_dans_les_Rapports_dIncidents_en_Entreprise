# Importation de la bibliothèque 
import fitz
import pandas as pd
import re
import spacy

# Charger le pdf contenant les informations

rapport = r"D:\Projects\Missions\Rappord d'incidents - Projet 3.pdf"

# Extraction des données textuelles dans un format txt
doc = fitz.open(rapport)
text = ""
for page in doc:
    text += page.get_text("text") + "\n"

doc.close()

with open(r'D:\Projects\IT\Data Science & IA\Detection_Emotions_Rapports_Incidents\Text_data\raport_incident.txt', "w", encoding="utf-8") as f :
    f.write(text)

# Importation des données textuelles
rapport = pd.read_csv(r"D:\Projects\IT\Data Science & IA\Detection_Emotions_Rapports_Incidents\Text_data\raport_incident.txt", delimiter = "\t", encoding="utf-8")

# Charger les textes
with open(r'D:\Projects\IT\Data Science & IA\Detection_Emotions_Rapports_Incidents\Text_data\raport_incident.txt', "r",encoding="utf-8") as f:
    texte = f.read()

# Enrégistrer mon texte en format csv
description = texte.split("\n")
incident = pd.DataFrame({'Description':description})
description.to_csv("incident.csv",index=False, encoding='utf-8')

# Prétraitement du texte avec Spacy
# Charger la nouvelle base de données
textrh = pd.read_excel(r'D:\Projects\IT\Data Science & IA\Detection_Emotions_Rapports_Incidents\Text_data\incident.xlsx')

# Charger le modèle SpaCy en français
nlp = spacy.load("fr_core_news_sm")

# Définition des stops_words
custom_stop_word = {
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "mon", "ton", "son", "ma", "ta", "sa", "notre", "votre", "leur",
    "moi", "lui", "nous", "eux", "elles", "est", "sont", "était", "étaient", "être", "été", "étant", 
    "un", "une", "le", "la", "les", "ce", "cette", "ces", "celui-ci", "celui-là", "et", "mais", "ou", "si", "parce que", "donc",
    "sur", "dans", "à", "par", "avec", "au sujet de", "contre", "entre", "dans", "à travers", "par-dessus", "sous",
    "encore", "davantage", "alors", "une fois", "peut", "va", "juste", "devrait", "ferait", "pourrait", "peut-être", "pourrait", "doit","inclure", "mentionner", "déjà", "rapidement", "bientôt", "permettre", "dehors", "deuxième", "loin",
    "marché", "million", "stock", "général", "industrie", "économie", "nation", "éducation",
    "moment", "parent", ""
}

# Nettoyage des espaces inutiles et caractères speciaux
def cleaning_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if (token.is_alpha or token.isspace()) and token.lemma_ not in custom_stop_word]
    return " ".join(tokens)

# Apllication
textrh['Inutilite_supr'] = textrh['Description'].dropna().astyp(str).apply(cleaning_text)

# Extraction des entités
def extraction_entity(text):
    doc = nlp(text)
    return  [(ent.text, ent.label_) for ent in doc.ents]

textrh["Entités"] = textrh['Inutilite_supr'].apply(extraction_entity)