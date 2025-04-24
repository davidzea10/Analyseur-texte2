import streamlit as st
import spacy
from spacy import displacy
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import fr_core_news_sm
nlp = fr_core_news_sm.load()

# Configuration de la page
st.set_page_config(page_title="Analyseur de Texte Intelligent", layout="wide")

# Sidebar
st.sidebar.title("⚙️ Options")
afficher_grammaire = st.sidebar.checkbox("Afficher l'analyse grammaticale", value=True)
afficher_entites = st.sidebar.checkbox("Afficher les entités nommées", value=True)
afficher_stats = st.sidebar.checkbox("Afficher les statistiques", value=True)
afficher_wordcloud = st.sidebar.checkbox("Afficher le nuage de mots", value=True)
afficher_sentiment = st.sidebar.checkbox("Simuler une analyse de sentiment", value=True)

# Pied de page déplacé dans la sidebar
st.sidebar.markdown("---")
st.sidebar.caption("Projet développé avec spaCy & Streamlit – Débuze et Moza 💡")

# Titre principal
st.title("🧠 Analyseur de Texte Intelligent")
st.markdown("Analyse grammaticale et détection des entités nommées d’un texte français avec spaCy.")

# Saisie de texte
st.subheader("✍️ Saisissez votre texte")
texte = st.text_area("Entrez un texte ici :", height=200, placeholder="Ex: Emmanuel Macron est né à Amiens...")

# Variable pour stocker les données à afficher
data = []

if st.button("🔍 Analyser") and texte:
    doc = nlp(texte)

    # Analyse grammaticale
    if afficher_grammaire:
        st.subheader("🧩 Analyse grammaticale")
        data = []
        for token in doc:
            data.append({
                "Texte": token.text,
                "Lemma": token.lemma_,
                "POS": token.pos_,
                "Dépendance": token.dep_,
            })
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

    # Entités nommées
    if afficher_entites:
        st.subheader("🏷️ Entités Nommées")
        if doc.ents:
            html = displacy.render(doc, style="ent")
            html = html.replace("#ffffff", "#ffffff")
            html = html.replace("color: black", "color: #444")
            st.components.v1.html(html, scrolling=True, height=250)
        else:
            st.info("Aucune entité nommée détectée.")

    # Statistiques
    if afficher_stats:
        st.subheader("📊 Statistiques")
        st.markdown(f"""
        - **Nombre total de tokens** : {len(doc)}
        - **Nombre de phrases** : {len(list(doc.sents))}
        - **Nombre d’entités** : {len(doc.ents)}
        """)

    # Nuage de mots
    if afficher_wordcloud:
        st.subheader("☁️ Nuage de mots")
        words = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
        if words:
            word_freq = Counter(words)
            wordcloud = WordCloud(width=800, height=300, background_color='white').generate_from_frequencies(word_freq)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.info("Pas assez de contenu pour générer un nuage de mots.")

    # Analyse de sentiment (simulation)
    if afficher_sentiment:
        st.subheader("💬 Sentiment simulé")
        mots_pos = ["bon", "excellent", "heureux", "positif"]
        mots_neg = ["mauvais", "triste", "négatif", "horrible"]
        score = 0
        for token in doc:
            if token.lemma_.lower() in mots_pos:
                score += 1
            elif token.lemma_.lower() in mots_neg:
                score -= 1

        if score > 0:
            st.success("Sentiment global : POSITIF")
        elif score < 0:
            st.error("Sentiment global : NÉGATIF")
        else:
            st.info("Sentiment global : NEUTRE")

else:
    st.warning("Entrez un texte puis cliquez sur 'Analyser'.")
