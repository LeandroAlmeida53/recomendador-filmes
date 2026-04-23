import streamlit as st
import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(os.path.join(nltk_data_path, "corpora", "stopwords")):
    nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="Recomenda Filmes", layout="centered")
st.title("🎬 Recomendador de Filmes")
st.markdown("Selecione um filme pra ver o que combina com ele.")

with open("filmes.json", "r", encoding="utf-8") as f:
    filmes_data = json.load(f)

df = pd.DataFrame(filmes_data)

stopwords_pt = stopwords.words('portuguese')

vectorizer = TfidfVectorizer(stop_words=stopwords_pt)
tfidf_mat = vectorizer.fit_transform(df['sinopse'])

sim_matrix = cosine_similarity(tfidf_mat)

@st.cache_resource
def get_recomendacoes(titulo_filme):
    idx = df.index[df['titulo'] == titulo_filme].tolist()[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores.sort(key=lambda x: x[1], reverse=True)
    top5_indices = [x[0] for x in scores[1:6]]
    return df.iloc[top5_indices]

filme = st.selectbox("Qual filme você gosta?", df['titulo'])

if st.button("Ver recomendações"):
    if not filme:
        st.warning("Seleciona um filme aí, vai...")
    else:
        try:
            recomendados = get_recomendacoes(filme)
            st.subheader(f"Quem curte {filme} também gosta de:")
            
            for _, rec in recomendados.iterrows():
                with st.expander(f"{rec['titulo']} ({rec['ano']})"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if rec['poster']:
                            st.image(rec['poster'], width=120)
                    with col2:
                        st.markdown(f"**Nota:** {rec['nota']}")
                        st.markdown(f"**Gêneros:** {', '.join(rec['generos'])}")
                        st.markdown(f"_{rec['sinopse']}_")
        except Exception:
            st.error("Deu algum problema na busca. Tenta outro filme?")