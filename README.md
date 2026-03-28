# 🎬 Recomendador de Filmes

Um aplicativo simples feito em Streamlit que recomenda filmes parecidos usando TF-IDF + similaridade de cosseno.

### Como rodar localmente

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
streamlit run app.py