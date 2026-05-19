import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0f0f1a; }
.block-container { padding: 1.5rem 2rem; max-width: 1100px; }

.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    margin-bottom: 24px;
    border: 1px solid #4c1d95;
}
.hero h1 { color: #fff; font-size: 2.2em; margin: 0 0 8px 0; }
.hero p  { color: #a78bfa; font-size: 1em; margin: 0; }

.movie-card {
    background: #1a1a2e;
    border: 1px solid #2d2d4e;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.movie-card:hover { border-color: #7c3aed; }
.movie-title { color: #e2e8f0; font-size: 1.05em; font-weight: 600; margin-bottom: 4px; }
.movie-meta  { color: #94a3b8; font-size: 0.82em; margin-bottom: 6px; }
.movie-overview { color: #cbd5e1; font-size: 0.82em; line-height: 1.5; }
.badge {
    display: inline-block;
    background: #312e81;
    color: #a78bfa;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75em;
    margin-right: 4px;
    margin-bottom: 4px;
}
.match-bar-wrap { margin: 6px 0 10px 0; }
.match-label { color: #7c3aed; font-size: 0.78em; margin-bottom: 3px; }
.bar-bg { background: #2d2d4e; border-radius: 6px; height: 10px; }
.bar-fill {
    background: linear-gradient(90deg, #7c3aed, #a78bfa);
    height: 10px; border-radius: 6px;
}
.rating-star { color: #fbbf24; font-size: 0.9em; }
.section-title { color: #a78bfa; font-size: 1.2em; font-weight: 700; margin: 20px 0 12px 0; }
.searched-card {
    background: linear-gradient(135deg,#1e1b4b,#1a1a2e);
    border: 1px solid #4c1d95;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 20px;
}
.searched-title { color: #fff; font-size: 1.3em; font-weight: 700; margin-bottom: 6px; }
.searched-meta  { color: #a78bfa; font-size: 0.85em; margin-bottom: 8px; }
.searched-overview { color: #cbd5e1; font-size: 0.85em; line-height: 1.55; }

/* Chart bars for browse */
.chart-bar-row { display:flex; align-items:center; gap:10px; margin:5px 0; }
.chart-label { width:180px; font-size:0.78em; color:#ddd; white-space:nowrap;
               overflow:hidden; text-overflow:ellipsis; }
.chart-bar-green { background:linear-gradient(90deg,#059669,#34d399);
                   height:18px; border-radius:4px; min-width:4px; }
.chart-bar-red   { background:linear-gradient(90deg,#dc2626,#f87171);
                   height:18px; border-radius:4px; min-width:4px; }
.chart-val { font-size:0.75em; color:#94a3b8; white-space:nowrap; }

div[data-testid="stTabs"] button { font-size:0.9em !important; }
div[data-testid="stVerticalBlock"] { gap: 0.4rem; }

@media (max-width: 640px) {
    .hero h1 { font-size: 1.5em; }
    .block-container { padding: 1rem; }
    .chart-label { width: 110px; }
}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading movies...")
def load_data():
    df = pd.read_csv("movies.csv")
    for col in ["genres","keywords","tagline","cast","director","overview"]:
        df[col] = df[col].fillna("")
    df["year"]     = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    df["runtime"]  = df["runtime"].fillna(0).astype(int)
    df["vote_count"]   = df["vote_count"].fillna(0).astype(int)
    df["vote_average"] = df["vote_average"].fillna(0)
    df["popularity"]   = df["popularity"].fillna(0)

    C = df["vote_average"].mean()
    m = df["vote_count"].quantile(0.6)
    df["weighted_score"] = (
        (df["vote_count"] / (df["vote_count"] + m)) * df["vote_average"]
        + (m / (df["vote_count"] + m)) * C
    )
    df["combined"] = (
        df["genres"]  + " " + df["keywords"] + " " + df["tagline"] + " " +
        df["cast"]    + " " + df["director"] + " " + df["overview"]
    )
    vec = TfidfVectorizer(max_features=10000)
    mat = vec.fit_transform(df["combined"])
    return df, mat

df, feature_matrix = load_data()
all_titles = df["title"].tolist()

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1>🎬 Movie Recommender</h1>
  <p>Discover movies similar to your favourites &nbsp;·&nbsp; 4,803 titles &nbsp;·&nbsp;
     Works on mobile &amp; PC</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Find Similar Movies", "🏆 Top Rated", "🔥 Most Popular", "ℹ️ About"])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Search
# ══════════════════════════════════════════════════════════════
with tab1:
    col_in, col_n = st.columns([4, 1])
    with col_in:
        movie_input = st.text_input("🎬 Enter a movie you like", placeholder="e.g. Inception, Avatar, The Dark Knight...")
    with col_n:
        num_results = st.number_input("Results", min_value=1, max_value=30, value=10, step=1)

    # Live suggestions
    if movie_input and len(movie_input) >= 2:
        suggestions = [t for t in all_titles if movie_input.lower() in t.lower()][:8]
        if suggestions:
            chosen = st.selectbox("💡 Did you mean?", ["— select —"] + suggestions)
            if chosen != "— select —":
                movie_input = chosen

    search_clicked = st.button("🎯 Find Similar Movies", type="primary", use_container_width=True)

    if search_clicked and movie_input:
        matches = difflib.get_close_matches(movie_input.strip(), all_titles, n=1, cutoff=0.4)
        if not matches:
            st.error(f"❌ No match found for **'{movie_input}'**. Try a different spelling.")
        else:
            matched_title = matches[0]
            movie_row     = df[df["title"] == matched_title].iloc[0]
            movie_idx     = int(movie_row["index"])

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
            yr  = str(movie_row["year"]) if movie_row["year"] > 0 else "N/A"
            gen = movie_row["genres"].replace(" ", " · ") or "N/A"
            cst = ", ".join(movie_row["cast"].split()[:5]) or "N/A"
            ov  = (movie_row["overview"][:300]+"...") if len(movie_row["overview"])>300 else movie_row["overview"]
            rt  = f"{int(movie_row['runtime'])} min" if movie_row["runtime"] > 0 else "N/A"
            st.markdown(f"""
            <div class='searched-card'>
              <div class='searched-title'>🎬 {matched_title}</div>
              <div class='searched-meta'>
                {yr} &nbsp;·&nbsp; {rt} &nbsp;·&nbsp; ⭐ {movie_row['vote_average']:.1f}/10
                &nbsp;·&nbsp; {int(movie_row['vote_count']):,} votes
              </div>
              <div style='margin-bottom:8px'>{gen}</div>
              <div style='color:#94a3b8;font-size:0.8em;margin-bottom:4px'><b>Cast:</b> {cst}</div>
              <div class='searched-overview'>{ov}</div>
            </div>
            """, unsafe_allow_html=True)

            # Bar chart
            st.markdown("<div class='section-title'>📊 Match Score Chart</div>", unsafe_allow_html=True)
            chart_html = ""
            for row, sim_score in results[:10]:
                t  = row["title"][:30]+"…" if len(row["title"])>30 else row["title"]
                pct = round(sim_score*100, 1)
                w   = max(4, int(pct * 5))
                chart_html += f"""
                <div class='chart-bar-row'>
                  <div class='chart-label' title='{row["title"]}'>{t}</div>
                  <div style='background:linear-gradient(90deg,#7c3aed,#a78bfa);
                       height:18px;border-radius:4px;min-width:4px;width:{w}px'></div>
                  <div class='chart-val'>{pct}% &nbsp; ★{row['vote_average']:.1f}</div>
                </div>"""
            st.markdown(f"<div style='background:#1a1a2e;border-radius:12px;padding:16px'>{chart_html}</div>",
                        unsafe_allow_html=True)

            # Movie cards
            st.markdown(f"<div class='section-title'>🎬 {len(results)} Similar Movies</div>", unsafe_allow_html=True)
            for i, (row, sim_score) in enumerate(results):
                yr2  = str(row["year"]) if row["year"] > 0 else "N/A"
                rt2  = f"{row['runtime']} min" if row["runtime"] > 0 else "N/A"
                gen2 = row["genres"].replace(" ", " · ") or "N/A"
                cst2 = ", ".join(row["cast"].split()[:4]) or "N/A"
                ov2  = (row["overview"][:180]+"...") if len(row["overview"])>180 else row["overview"]
                pct  = round(sim_score*100, 1)
                w    = max(2, int(pct*5))
                st.markdown(f"""
                <div class='movie-card'>
                  <div class='movie-title'>{i+1}. {row['title']}</div>
                  <div class='movie-meta'>
                    {yr2} &nbsp;·&nbsp; {rt2} &nbsp;·&nbsp;
                    <span class='rating-star'>⭐ {row['vote_average']:.1f}/10</span>
                  </div>
                  <div style='margin-bottom:6px'>{gen2}</div>
                  <div class='match-bar-wrap'>
                    <div class='match-label'>Match: {pct}%</div>
                    <div class='bar-bg'><div class='bar-fill' style='width:{w}%'></div></div>
                  </div>
                  <div style='color:#94a3b8;font-size:0.78em;margin-bottom:4px'><b>Cast:</b> {cst2}</div>
                  <div class='movie-overview'>{ov2}</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — Top Rated
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🏆 Top Rated Movies")
    all_genres = set()
    for g in df["genres"].dropna():
        for word in g.split():
            if len(word) > 2: all_genres.add(word)
    genre_list = ["All"] + sorted(all_genres)

    col_g, col_c = st.columns([2,1])
    with col_g:
        genre_top = st.selectbox("Filter by Genre", genre_list, key="gtop")
    with col_c:
        count_top = st.number_input("How many", 5, 50, 20, 1, key="ctop")

    if st.button("▶ Load Top Rated", type="primary", use_container_width=True, key="btop"):
        filtered = df[df["vote_count"] >= 100].copy()
        if genre_top != "All":
            filtered = filtered[filtered["genres"].str.contains(genre_top, case=False, na=False)]
        top = filtered.nlargest(int(count_top), "weighted_score")
        max_score = top["weighted_score"].max()

        # Chart
        chart_html = ""
        for _, r in top.iterrows():
            t = r["title"][:28]+"…" if len(r["title"])>28 else r["title"]
            w = max(4, int((r["weighted_score"]/max_score)*360))
            chart_html += f"""
            <div class='chart-bar-row'>
              <div class='chart-label' title='{r["title"]}'>{t}</div>
              <div class='chart-bar-green' style='width:{w}px'></div>
              <div class='chart-val'>★{r['vote_average']:.1f} &nbsp; ({int(r['vote_count']):,} votes)</div>
            </div>"""
        st.markdown(f"""
        <div style='background:#1a1a2e;border-radius:12px;padding:16px;margin-bottom:16px'>
          <div style='color:#34d399;font-weight:700;margin-bottom:10px'>📊 Weighted Rating Chart</div>
          {chart_html}
        </div>""", unsafe_allow_html=True)

        # Cards
        for i, (_, r) in enumerate(top.iterrows()):
            yr  = str(r["year"]) if r["year"] > 0 else "N/A"
            gen = r["genres"].replace(" "," · ") or "N/A"
            ov  = (r["overview"][:160]+"...") if len(r["overview"])>160 else r["overview"]
            st.markdown(f"""
            <div class='movie-card'>
              <div class='movie-title'>{i+1}. {r['title']}</div>
              <div class='movie-meta'>{yr} &nbsp;·&nbsp; ⭐ {r['vote_average']:.1f}/10
                &nbsp;·&nbsp; {int(r['vote_count']):,} votes &nbsp;·&nbsp; Dir: {r['director'] or 'N/A'}</div>
              <div style='margin-bottom:6px'>{gen}</div>
              <div class='movie-overview'>{ov}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — Most Popular
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔥 Most Popular Movies")
    col_g2, col_c2 = st.columns([2,1])
    with col_g2:
        genre_pop = st.selectbox("Filter by Genre", ["All"] + sorted(all_genres), key="gpop")
    with col_c2:
        count_pop = st.number_input("How many", 5, 50, 20, 1, key="cpop")

    if st.button("▶ Load Popular Movies", type="primary", use_container_width=True, key="bpop"):
        filtered2 = df.copy()
        if genre_pop != "All":
            filtered2 = filtered2[filtered2["genres"].str.contains(genre_pop, case=False, na=False)]
        top2    = filtered2.nlargest(int(count_pop), "popularity")
        max_pop = top2["popularity"].max()

        chart_html2 = ""
        for _, r in top2.iterrows():
            t = r["title"][:28]+"…" if len(r["title"])>28 else r["title"]
            w = max(4, int((r["popularity"]/max_pop)*360))
            chart_html2 += f"""
            <div class='chart-bar-row'>
              <div class='chart-label' title='{r["title"]}'>{t}</div>
              <div class='chart-bar-red' style='width:{w}px'></div>
              <div class='chart-val'>{r['popularity']:.0f} pts &nbsp; ★{r['vote_average']:.1f}</div>
            </div>"""
        st.markdown(f"""
        <div style='background:#1a1a2e;border-radius:12px;padding:16px;margin-bottom:16px'>
          <div style='color:#f87171;font-weight:700;margin-bottom:10px'>📊 Popularity Score Chart</div>
          {chart_html2}
        </div>""", unsafe_allow_html=True)

        for i, (_, r) in enumerate(top2.iterrows()):
            yr  = str(r["year"]) if r["year"] > 0 else "N/A"
            gen = r["genres"].replace(" "," · ") or "N/A"
            ov  = (r["overview"][:160]+"...") if len(r["overview"])>160 else r["overview"]
            st.markdown(f"""
            <div class='movie-card'>
              <div class='movie-title'>{i+1}. {r['title']}</div>
              <div class='movie-meta'>{yr} &nbsp;·&nbsp; ⭐ {r['vote_average']:.1f}/10
                &nbsp;·&nbsp; Popularity: {r['popularity']:.0f}</div>
              <div style='margin-bottom:6px'>{gen}</div>
              <div class='movie-overview'>{ov}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — About
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"""
## 🎬 About Movie Recommender

**Movie Recommender** helps you instantly find films similar to the ones you love —
no account, no sign-up, works on any device.

---

### 🧠 How recommendations work

| Step | What happens |
|------|-------------|
| 1 | Each movie's genres, keywords, cast, director, tagline & overview are combined into one text |
| 2 | **TF-IDF** converts that text into a numerical vector |
| 3 | **Cosine Similarity** measures how close two movie vectors are — closer = more similar |
| 4 | **Fuzzy matching** handles typos so you don't need perfect spelling |
| 5 | Results are ranked by similarity with ratings, runtime & overview shown |

### 📊 Browse modes
- **Top Rated** — Bayesian weighted score (balances rating + number of votes)
- **Popular** — TMDB popularity index based on views, watchlist adds & rating activity

### 📂 Dataset
**{len(df):,} movies** from the TMDB 5000 Movie Dataset

### 🛠️ Built with
Python · Scikit-learn · Pandas · Streamlit
    """)
