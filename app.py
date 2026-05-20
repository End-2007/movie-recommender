import streamlit as st
import pandas as pd
import numpy as np
import difflib
import ast
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding: 1.5rem 2rem; max-width: 1100px; }
.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
    border-radius: 16px; padding: 32px 24px; text-align: center;
    margin-bottom: 24px; border: 1px solid #4c1d95;
}
.hero h1 { color: #fff; font-size: 2.2em; margin: 0 0 8px 0; }
.hero p  { color: #a78bfa; font-size: 1em; margin: 0; }
.movie-card {
    background: #1a1a2e; border: 1px solid #2d2d4e;
    border-radius: 12px; padding: 16px; margin-bottom: 10px;
}
.movie-title    { color: #e2e8f0; font-size: 1.05em; font-weight: 600; margin-bottom: 4px; }
.movie-meta     { color: #94a3b8; font-size: 0.82em; margin-bottom: 6px; }
.movie-overview { color: #cbd5e1; font-size: 0.82em; line-height: 1.5; }
.searched-card {
    background: linear-gradient(135deg,#1e1b4b,#1a1a2e);
    border: 1px solid #4c1d95; border-radius: 12px; padding: 18px; margin-bottom: 16px;
}
footer { display: none !important; }
@media (max-width:640px) {
    .hero h1 { font-size: 1.5em; }
    .block-container { padding: 1rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Language map ──────────────────────────────────────────────
LANG_MAP = {
    "All":              "All",
    "English":          "en",
    "Hindi":            "hi",
    "Tamil":            "ta",
    "Telugu":           "te",
    "Malayalam":        "ml",
    "Kannada":          "kn",
    "Bengali":          "bn",
    "Marathi":          "mr",
    "Punjabi":          "pa",
    "Korean":           "ko",
    "Japanese":         "ja",
    "French":           "fr",
    "Spanish":          "es",
    "German":           "de",
    "Chinese":          "zh",
    "Italian":          "it",
}

# ── Load data ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Loading movies...")
def load_data():
    df = pd.read_csv("movies.csv")
    for col in ["genres","keywords","tagline","cast","director","overview","original_language"]:
        df[col] = df[col].fillna("")
    df["year"]         = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    df["runtime"]      = df["runtime"].fillna(0).astype(int)
    df["vote_count"]   = df["vote_count"].fillna(0).astype(int)
    df["vote_average"] = df["vote_average"].fillna(0)
    df["popularity"]   = df["popularity"].fillna(0)

    def parse_countries(val):
        try:
            items = ast.literal_eval(str(val))
            return [i["name"] for i in items] if isinstance(items, list) else []
        except:
            return []
    df["countries"]     = df["production_countries"].apply(parse_countries)
    df["countries_str"] = df["countries"].apply(lambda x: ", ".join(x[:2]))

    C = df["vote_average"].mean()
    m = df["vote_count"].quantile(0.6)
    df["weighted_score"] = (
        (df["vote_count"] / (df["vote_count"] + m)) * df["vote_average"]
        + (m / (df["vote_count"] + m)) * C
    )
    df["combined"] = (
        df["genres"] + " " + df["keywords"] + " " + df["tagline"] + " " +
        df["cast"]   + " " + df["director"] + " " + df["overview"]
    )
    vec = TfidfVectorizer(max_features=10000)
    mat = vec.fit_transform(df["combined"])
    return df, mat

df, feature_matrix = load_data()
all_titles = df["title"].tolist()

all_genres = set()
for g in df["genres"].dropna():
    for word in g.split():
        if len(word) > 2: all_genres.add(word)
genre_list = ["All"] + sorted(all_genres)

all_countries = set()
for c in df["countries"]:
    all_countries.update(c)
country_list = ["All"] + sorted(all_countries)


# ── Chart functions (Plotly — renders perfectly) ──────────────
def make_bar_chart(titles, values, ratings, color, title, x_label):
    short_titles = [t[:30]+"…" if len(t)>30 else t for t in titles]
    hover = [f"<b>{t}</b><br>{x_label}: {v:.1f}<br>⭐ Rating: {r:.1f}/10"
             for t,v,r in zip(titles, values, ratings)]
    fig = go.Figure(go.Bar(
        x=values,
        y=short_titles,
        orientation="h",
        marker=dict(
            color=values,
            colorscale=color,
            showscale=False,
            line=dict(width=0)
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont=dict(color="white", size=11),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=14)),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="white"),
        height=max(350, len(titles) * 38),
        margin=dict(l=10, r=60, t=40, b=20),
        xaxis=dict(
            showgrid=True, gridcolor="#2d2d4e",
            tickfont=dict(color="#94a3b8"),
            title=dict(text=x_label, font=dict(color="#94a3b8")),
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="#e2e8f0", size=11),
            autorange="reversed",
        ),
        hoverlabel=dict(bgcolor="#312e81", font_color="white"),
    )
    return fig


# ── Movie card ────────────────────────────────────────────────
def movie_card(i, row, extra_badge=""):
    yr   = str(row["year"]) if row["year"] > 0 else "N/A"
    rt   = f"{row['runtime']} min" if row["runtime"] > 0 else "N/A"
    gen  = row["genres"].replace(" ", " · ") or "N/A"
    cst  = ", ".join(row["cast"].split()[:4]) or "N/A"
    ov   = (row["overview"][:200]+"...") if len(row["overview"])>200 else row["overview"]
    lang = row["original_language"].upper()
    ctry = row["countries_str"] or "N/A"
    num  = f"{i}. " if i else ""
    st.markdown(f"""
    <div class='movie-card'>
      <div class='movie-title'>{num}{row['title']}</div>
      <div class='movie-meta'>
        {yr} &nbsp;·&nbsp; {rt} &nbsp;·&nbsp; ⭐ {row['vote_average']:.1f}/10
        &nbsp;·&nbsp; {int(row['vote_count']):,} votes
        &nbsp;·&nbsp; <b>{lang}</b> &nbsp;·&nbsp; {ctry}
        {extra_badge}
      </div>
      <div style='color:#7c3aed;font-size:0.78em;margin-bottom:6px'>{gen}</div>
      <div style='color:#94a3b8;font-size:0.78em;margin-bottom:5px'><b>Cast:</b> {cst}</div>
      <div class='movie-overview'>{ov}</div>
    </div>""", unsafe_allow_html=True)


def filter_df(data, lang_key, country_key, genre_key):
    f = data.copy()
    lang_val = LANG_MAP.get(lang_key, "All")
    if lang_val != "All":
        f = f[f["original_language"] == lang_val]
    if country_key != "All":
        f = f[f["countries"].apply(lambda x: country_key in x)]
    if genre_key != "All":
        f = f[f["genres"].str.contains(genre_key, case=False, na=False)]
    return f


# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1>🎬 Movie Recommender</h1>
  <p>Find movies similar to your favourites &nbsp;·&nbsp; 9,000+ titles &nbsp;·&nbsp;
     Hindi · Tamil · Telugu · Korean · Japanese · French & more</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Find Similar Movies", "🏆 Top Rated", "🔥 Most Popular", "ℹ️ About"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Search
# ══════════════════════════════════════════════════════════════
with tab1:
    movie_input = st.text_input(
        "🎬 Enter a movie you like",
        placeholder="e.g. RRR, Inception, Parasite, Dilwale, Spirited Away..."
    )

    if movie_input and len(movie_input) >= 2:
        suggestions = [t for t in all_titles if movie_input.lower() in t.lower()][:8]
        if suggestions:
            chosen = st.selectbox("💡 Did you mean?", ["— select —"] + suggestions)
            if chosen != "— select —":
                movie_input = chosen

    num_results = st.slider("Number of recommendations", 1, 30, 10, 1)
    search_clicked = st.button("🎯 Find Similar Movies", type="primary", use_container_width=True)

    if search_clicked and movie_input:
        matches = difflib.get_close_matches(movie_input.strip(), all_titles, n=1, cutoff=0.3)
        if not matches:
            st.error(f"❌ No match found for **'{movie_input}'**. Try a different spelling.")
        else:
            matched_title = matches[0]
            movie_row     = df[df["title"] == matched_title].iloc[0]
            movie_idx     = int(movie_row["index"])

            with st.spinner("Finding similar movies..."):
                sim_vector = cosine_similarity(feature_matrix[movie_idx], feature_matrix)[0]
                sim_scores = sorted(enumerate(sim_vector), key=lambda x: x[1], reverse=True)

            results, seen = [], set()
            for raw_idx, sim_score in sim_scores:
                if raw_idx >= len(df): continue
                row = df.iloc[raw_idx]
                if row["title"] == matched_title or row["title"] in seen: continue
                seen.add(row["title"])
                results.append((row, sim_score))
                if len(results) >= num_results: break

            # Searched movie card
            yr   = str(movie_row["year"]) if movie_row["year"] > 0 else "N/A"
            gen  = movie_row["genres"].replace(" ", " · ") or "N/A"
            cst  = ", ".join(movie_row["cast"].split()[:5]) or "N/A"
            ov   = (movie_row["overview"][:300]+"...") if len(movie_row["overview"])>300 else movie_row["overview"]
            rt   = f"{int(movie_row['runtime'])} min" if movie_row["runtime"] > 0 else "N/A"
            lang = movie_row["original_language"].upper()
            ctry = movie_row["countries_str"] or "N/A"
            st.markdown(f"""
            <div class='searched-card'>
              <div style='color:#fff;font-size:1.3em;font-weight:700;margin-bottom:6px'>🎬 {matched_title}</div>
              <div style='color:#a78bfa;font-size:0.85em;margin-bottom:8px'>
                {yr} &nbsp;·&nbsp; {rt} &nbsp;·&nbsp; ⭐ {movie_row['vote_average']:.1f}/10
                &nbsp;·&nbsp; {int(movie_row['vote_count']):,} votes
                &nbsp;·&nbsp; <b>{lang}</b> &nbsp;·&nbsp; {ctry}
              </div>
              <div style='color:#7c3aed;font-size:0.85em;margin-bottom:8px'>{gen}</div>
              <div style='color:#94a3b8;font-size:0.8em;margin-bottom:4px'><b>Cast:</b> {cst}</div>
              <div style='color:#cbd5e1;font-size:0.85em;line-height:1.55'>{ov}</div>
            </div>""", unsafe_allow_html=True)

            # ── Bar chart FIRST ────────────────────────────────
            chart_titles  = [row["title"] for row, _ in results]
            chart_scores  = [round(s*100, 1) for _, s in results]
            chart_ratings = [row["vote_average"] for row, _ in results]

            fig = make_bar_chart(
                chart_titles, chart_scores, chart_ratings,
                color="Purples",
                title=f"📊 Similarity Match — Top {len(results)} Movies similar to '{matched_title}'",
                x_label="Match Score (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Movie cards after chart ────────────────────────
            st.markdown(f"<div style='color:#a78bfa;font-size:1.1em;font-weight:700;margin:16px 0 8px'>🎬 {len(results)} Recommended Movies</div>",
                        unsafe_allow_html=True)
            for i, (row, sim_score) in enumerate(results):
                pct = round(sim_score*100, 1)
                movie_card(i+1, row, extra_badge=f"&nbsp;·&nbsp; 🎯 Match: {pct}%")


# ══════════════════════════════════════════════════════════════
# TAB 2 — Top Rated
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🏆 Top Rated Movies")
    c1, c2, c3 = st.columns(3)
    with c1: lang_top    = st.selectbox("Language", list(LANG_MAP.keys()), key="ltop")
    with c2: country_top = st.selectbox("Country",  country_list,          key="cotop")
    with c3: genre_top   = st.selectbox("Genre",    genre_list,            key="gtop")
    count_top = st.slider("How many to show", 5, 50, 15, 1, key="ctop")

    if st.button("▶ Load Top Rated", type="primary", use_container_width=True, key="btop"):
        filtered = filter_df(df[df["vote_count"] >= 50], lang_top, country_top, genre_top)
        if filtered.empty:
            st.warning("No movies found. Try different filters.")
        else:
            top = filtered.nlargest(int(count_top), "weighted_score")

            # Chart first
            fig2 = make_bar_chart(
                top["title"].tolist(),
                top["weighted_score"].round(2).tolist(),
                top["vote_average"].tolist(),
                color="Greens",
                title="📊 Top Rated — Weighted Score (balances rating + number of votes)",
                x_label="Weighted Score"
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Cards
            st.markdown("<div style='color:#34d399;font-size:1.1em;font-weight:700;margin:10px 0 8px'>🏆 Movies</div>",
                        unsafe_allow_html=True)
            for i, (_, r) in enumerate(top.iterrows()):
                movie_card(i+1, r, extra_badge=f"&nbsp;·&nbsp; Dir: {r['director'] or 'N/A'}")


# ══════════════════════════════════════════════════════════════
# TAB 3 — Most Popular
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔥 Most Popular Movies")
    c4, c5, c6 = st.columns(3)
    with c4: lang_pop    = st.selectbox("Language", list(LANG_MAP.keys()), key="lpop")
    with c5: country_pop = st.selectbox("Country",  country_list,          key="copop")
    with c6: genre_pop   = st.selectbox("Genre",    genre_list,            key="gpop")
    count_pop = st.slider("How many to show", 5, 50, 15, 1, key="cpop")

    if st.button("▶ Load Popular Movies", type="primary", use_container_width=True, key="bpop"):
        filtered2 = filter_df(df, lang_pop, country_pop, genre_pop)
        if filtered2.empty:
            st.warning("No movies found. Try different filters.")
        else:
            top2 = filtered2.nlargest(int(count_pop), "popularity")

            # Chart first
            fig3 = make_bar_chart(
                top2["title"].tolist(),
                top2["popularity"].round(1).tolist(),
                top2["vote_average"].tolist(),
                color="Reds",
                title="📊 Most Popular — Popularity Score (views, watchlists, ratings activity)",
                x_label="Popularity Score"
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Cards
            st.markdown("<div style='color:#f87171;font-size:1.1em;font-weight:700;margin:10px 0 8px'>🔥 Movies</div>",
                        unsafe_allow_html=True)
            for i, (_, r) in enumerate(top2.iterrows()):
                movie_card(i+1, r, extra_badge=f"&nbsp;·&nbsp; Popularity: {r['popularity']:.0f}")


# ══════════════════════════════════════════════════════════════
# TAB 4 — About
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"""
## 🎬 About Movie Recommender

**Movie Recommender** helps you instantly find films similar to ones you love —
no account needed, works on any device.

---
### 🧠 How recommendations work
| Step | What happens |
|------|-------------|
| 1 | Each movie's genres, keywords, cast, director, tagline & overview are combined |
| 2 | **TF-IDF** converts that text into numerical vectors |
| 3 | **Cosine Similarity** finds the closest matching movies |
| 4 | **Fuzzy matching** handles typos in movie names |

### 🌍 Languages supported
Hindi · Tamil · Telugu · Malayalam · Kannada · Bengali · Marathi · Punjabi ·
Korean · Japanese · French · Spanish · German · Chinese · Italian · English

### 📊 Browse modes
- **Top Rated** — Bayesian weighted score (balances high rating + enough votes)
- **Most Popular** — TMDB popularity index

### 📂 Dataset
**{len(df):,} movies** across 16+ languages from the TMDB database

### 🛠️ Built with
Python · Scikit-learn · Pandas · Plotly · Streamlit
    """)
