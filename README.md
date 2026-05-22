# 🎬 Movie Recommender

A content-based movie recommendation web app built with Python and Streamlit — supporting **9,000+ movies across 16+ languages** including Hindi, Tamil, Telugu, Korean, Japanese, French and more.

🔗 **Live App:** [movie-recommender-xybqrahokudiru8mzmabjj.streamlit.app](https://movie-recommender-xybqrahokudiru8mzmabjj.streamlit.app/)

---

## Features

- **Find Similar Movies** — type any movie name and get top recommendations with match scores
- **Live Autocomplete** — suggestions appear as you type
- **Interactive Bar Charts** — visualize similarity scores, ratings and popularity using Plotly
- **Top Rated** — browse highest-rated movies with genre, language and country filters
- **Most Popular** — explore trending movies by popularity score
- **Multi-language Support** — Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, Korean, Japanese, French, Spanish, German, Chinese, Italian, English and more
- **Works on Mobile & PC** — fully responsive layout

---

## How It Works

1. Each movie's genres, keywords, cast, director, tagline and overview are combined into a single text
2. **TF-IDF Vectorization** converts that text into numerical vectors — rare but meaningful words get higher weight
3. **Cosine Similarity** measures how close two movie vectors are — closer angle = more similar movies
4. **Fuzzy Matching** (via `difflib`) handles typos in movie names
5. Browse tabs use a **Bayesian Weighted Score** (balances rating + vote count) for fair rankings

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Web App | Streamlit |
| ML / Similarity | Scikit-learn (TF-IDF, Cosine Similarity) |
| Data Processing | Pandas, NumPy |
| Visualizations | Plotly |
| Fuzzy Matching | difflib |
| Dataset | TMDB 5000 + TMDB API |

---

## Dataset

- **9,058 movies** sourced from the TMDB 5000 Movie Dataset and TMDB API
- Features used: `genres`, `keywords`, `cast`, `director`, `tagline`, `overview`, `original_language`, `production_countries`

---

## Project Structure

```
movie-recommender/
├── app.py              # Main Streamlit application
├── movies.csv          # Dataset (9,058 movies)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/End-2007/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
streamlit run app.py
```

---

## Languages Supported

Hindi · Tamil · Telugu · Malayalam · Kannada · Bengali · Marathi · Punjabi · Korean · Japanese · French · Spanish · German · Chinese · Italian · English
