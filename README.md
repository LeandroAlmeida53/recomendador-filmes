# 🎬 Recomendador de Filmes

Um aplicativo de recomendação de filmes baseado em similaridade de texto usando TF-IDF + cosine similarity.

### Testando o app

Acessar Recomendador de Filmes: (https://recomendador-filmes.streamlit.app)

*(Basta escolher um filme e clicar em "Ver recomendações". As sugestões mudam conforme o filme selecionado.)*

### Tecnologias utilizadas

- **Streamlit** – Interface web interativa
- **scikit-learn** – Vetorização TF-IDF e cálculo de similaridade
- **NLTK** – Stopwords em português
- **Pandas** – Manipulação dos dados de filmes
- **TMDB API** – Dados dos filmes (título, sinopse, poster, gêneros, notas)

### Dados

Os dados dos filmes foram obtidos dalla TMDB API (The Movie Database). O app utiliza um arquivo JSON cacheado com mais de 100 filmes populares para cálculo de similaridade.

### Sobre o projeto

Sistema de recomendação de conteúdo usando processamento de linguagem natural.
Projeto desenvolvido para prática de Data Science e deploy em nuvem.

---

*Este projeto usa dados da TMDB. Os dados pertenecen à The Movie Database.*