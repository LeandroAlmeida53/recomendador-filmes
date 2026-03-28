import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Baixa stopwords automaticamente se não existir (importante pro Streamlit Cloud)
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(os.path.join(nltk_data_path, "corpora", "stopwords")):
    nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="Recomenda Filmes", layout="centered")
st.title("🎬 Recomendador de Filmes")
st.markdown("Selecione um filme pra ver o que combina com ele.")

# Dados
filmes = {
    'titulo': [
        "Matrix", "Matrix Reloaded", "Matrix Revolutions",
        "Senhor dos Anéis", "O Hobbit", "Vingadores",
        "Homem de Ferro", "Capitão América", "Interestelar",
        "Deadpool", "Deadpool 2", "Logan"
    ],
    'descricao': [
        "Programador vira rebelde ao descobrir que o mundo inteiro é uma simulação controlada por máquinas inteligentes.",
        "Neo e a tripulação da Nabucodonosor enfrentam novos agentes e dilemas dentro da Matrix.",
        "A guerra definitiva entre humanos e máquinas chega ao fim com Neo como a última esperança.",
        "Frodo Bolseiro sai de Bolsão para destruir o Um Anel antes que Sauron domine a Terra-média.",
        "Bilbo Bolseiro é arrastado para uma grande aventura com 13 anões e o mago Gandalf.",
        "Os Vingadores se reúnem pela primeira vez para lutar contra Loki e os Chitauri.",
        "Tony Stark, gênio bilionário, constrói a primeira armadura do Homem de Ferro depois de ser sequestrado.",
        "Steve Rogers, um soldado franzino, vira o Capitão América com um soro experimental na Segunda Guerra.",
        "Cooper, ex-piloto da NASA, entra num buraco de minhoca para salvar a humanidade da extinção.",
        "Deadpool, o mercenário boca-suja com fator de cura, usa piadas ruins e katanas para acabar com bandidos.",
        "Deadpool forma o X-Force para proteger uma jovem mutante de uma empresa maligna.",
        "Wolverine, já velho e cansado, protege uma garota mutante enquanto foge de caçadores no futuro."
    ]
}

df = pd.DataFrame(filmes)

stopwords_pt = stopwords.words('portuguese')

vectorizer = TfidfVectorizer(stop_words=stopwords_pt)
tfidf_mat = vectorizer.fit_transform(df['descricao'])

sim_matrix = cosine_similarity(tfidf_mat)

def get_recomendacoes(titulo_filme):
    idx = df.index[df['titulo'] == titulo_filme].tolist()[0]
    scores = list(enumerate(sim_matrix[idx]))
    scores.sort(key=lambda x: x[1], reverse=True)
    top5_indices = [x[0] for x in scores[1:6]]
    return df['titulo'].iloc[top5_indices].values


# Interface
filme = st.selectbox("Qual filme você gosta?", df['titulo'])

if st.button("Ver recomendações"):
    if not filme:
        st.warning("Seleciona um filme aí, vai...")
    else:
        try:
            recomendados = get_recomendacoes(filme)
            st.subheader(f"Quem curte {filme} também gosta de:")
            for rec in recomendados:
                st.markdown(f"→ {rec}")
        except Exception:
            st.error("Deu algum problema na busca. Tenta outro filme?")