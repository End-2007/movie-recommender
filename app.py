import streamlit as st
import pandas as pd
import numpy as np
import difflib
import ast
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
.movie-title   { color: #e2e8f0; font-size: 1.05em; font-weight: 600; margin-bottom: 4px; }
.movie-meta    { color: #94a3b8; font-size: 0.82em; margin-bottom: 6px; }
.movie-overview{ color: #cbd5e1; font-size: 0.82em; line-height: 1.5; }
.bar-bg  { background: #2d2d4e; border-radius: 6px; height: 10px; width: 100%; }
.bar-fill{ height: 10px; border-radius: 6px;
           background: linear-gradient(90deg, #7c3aed, #a78bfa); }
.searched-card {
    background: linear-gradient(135deg,#1e1b4b,#1a1a2e);
    border: 1px solid #4c1d95; border-radius: 12px; padding: 18px; margin-bottom: 20px;
}
.chart-row { display:flex; align-items:center; gap:10px; margin:5px 0; }
.chart-label { width:180px; font-size:0.78em; color:#ddd;
               white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.chart-val { font-size:0.75em; color:#94a3b8; white-space:nowrap; }
footer { display:none !important; }
@media (max-width:640px) {
    .hero h1 { font-size:1.5em; }
    .block-container { padding:1rem; }
    .chart-label { width:110px; }
}
</style>
""", unsafe_allow_html=True)

# ── Language map ──────────────────────────────────────────────
LANG_MAP = {
    "All": "All",
    "English (en)": "en",
    "Hindi (hi)": "hi",
    "French (fr)": "fr",
    "Spanish (es)": "es",
    "German (de)": "de",
    "Chinese (zh/cn)": ["zh","cn"],
    "Japanese (ja)": "ja",
    "Korean (ko)": "ko",
    "Italian (it)": "it",
    "Russian (ru)": "ru",
    "Portuguese (pt)": "pt",
}

# ── Load data ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="🎬 Loading movies...")
def load_data():
    df = pd.read_csv("movies.csv")
    for col in ["genres","keywords","tagline","cast","director","overview","original_language"]:
        df[col] = df[col].fillna("")

    # Parse production countries
    def parse_countries(val):
        try:
            items = ast.literal_eval(val)
            return [i["name"] for i in items] if isinstance(items, list) else []
        except:
            return []
    df["countries"] = df["production_countries"].apply(parse_countries)
    df["countries_str"] = df["countries"].apply(lambda x: ", ".join(x))

    df["year"]       = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    df["runtime"]    = df["runtime"].fillna(0).astype(int)
    df["vote_count"] = df["vote_count"].fillna(0).astype(int)
    df["vote_average"] = df["vote_average"].fillna(0)
    df["popularity"]   = df["popularity"].fillna(0)

    C = df["vote_average"].mean()
    m = df["vote_count"].quantile(0.6)
    df["weighted_score"] = (
        (df["vote_count"]/(df["vote_count"]+m)) * df["vote_average"]
        + (m/(df["vote_count"]+m)) * C
    )
    df["combined"] = (
        df["genres"]+" "+df["keywords"]+" "+df["tagline"]+" "+
        df["cast"]+" "+df["director"]+" "+df["overview"]
    )
    vec = TfidfVectorizer(max_features=10000)
    mat = vec.fit_transform(df["combined"])
    return df, mat

df, feature_matrix = load_data()
all_titles = df["title"].tolist()

# All countries
all_countries = set()
for c in df["countries"]:
    all_countries.update(c)
country_list = ["All"] + sorted(all_countries)

# All genres
all_genres = set()
for g in df["genres"].dropna():
    for word in g.split():
        if len(word) > 2: all_genres.add(word)
genre_list = ["All"] + sorted(all_genres)

# ── Helpers ───────────────────────────────────────────────────
def movie_card(i, title, year, runtime, rating, votes, genres, cast, overview, extra=""):
    rt  = f"{runtime} min" if runtime > 0 else "N/A"
    yr  = str(year) if year > 0 else "N/A"
    gen = genres.replace(" ", " · ") or "N/A"
    cst = ", ".join(cast.split()[:4]) or "N/A"
    ov  = (overview[:180]+"...") if len(overview)>180 else overview
    num = f"{i}. " if i else ""
    st.markdown(f"""
    <div class='movie-card'>
      <div class='movie-title'>{num}{title}</div>
      <div class='movie-meta'>{yr} &nbsp;·&nbsp; {rt} &nbsp;·&nbsp;
        ⭐ {rating:.1f}/10 &nbsp;·&nbsp; {int(votes):,} votes {extra}</div>
      <div style='color:#7c3aed;font-size:0.78em;margin-bottom:6px'>{gen}</div>
      <div style='color:#94a3b8;font-size:0.78em;margin-bottom:4px'><b>Cast:</b> {cst}</div>
      <div class='movie-overview'>{ov}</div>
    </div>""", unsafe_allow_html=True)

def bar_chart(items, color1, color2, label_key, bar_key, val_text_fn, title):
    max_val = max(r[bar_key] for r in items) if items else 1
    rows = ""
    for r in items:
        t = r[label_key][:28]+"…" if len(r[label_key])>28 else r[label_key]
        w = max(4, int((r[bar_key]/max_val)*360))
        rows += f"""
        <div class='chart-row'>
          <div class='chart-label' title='{r[label_key]}'>{t}</div>
          <div style='background:linear-gradient(90deg,{color1},{color2});
               height:18px;border-radius:4px;min-width:4px;width:{w}px'></div>
          <div class='chart-val'>{val_text_fn(r)}</div>
        </div>"""
    st.markdown(f"""
    <div style='background:#1a1a2e;border-radius:12px;padding:16px;margin-bottom:16px'>
      <div style='color:{color2};font-weight:700;margin-bottom:10px'>{title}</div>
      {rows}
    </div>""", unsafe_allow_html=True)

def filter_by_lang_country(data, lang_key, country_key):
    filtered = data.copy()
    # Language filter
    lang_val = LANG_MAP.get(lang_key, "All")
    if lang_val != "All":
        if isinstance(lang_val, list):
            filtered = filtered[filtered["original_language"].isin(lang_val)]
        else:
            filtered = filtered[filtered["original_language"] == lang_val]
    # Country filter
    if country_key != "All":
        filtered = filtered[filtered["countries"].apply(lambda x: country_key in x)]
    return filtered

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1>🎬 Movie Recommender</h1>
  <p>Discover movies similar to your favourites &nbsp;·&nbsp; 4,803 titles &nbsp;·&nbsp;
     Multi-language &nbsp;·&nbsp; Works on mobile &amp; PC</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Find Similar", "🏆 Top Rated", "🔥 Most Popular", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Search
# ══════════════════════════════════════════════════════════════
with tab1:
    movie_input = st.text_input("🎬 Enter a movie you like",
        placeholder="e.g. Inception, Avatar, Dilwale Dulhania Le Jayenge...")

    if movie_input and len(movie_input) >= 2:
        suggestions = [t for t in all_titles if movie_input.lower() in t.lower()][:8]
        if suggestions:
            chosen = st.selectbox("💡 Did you mean?", ["— select —"] + suggestions)
            if chosen != "— select —":
                movie_input = chosen

    num_results = st.slider("Number of recommendations", 1, 30, 10, 1)
    search_clicked = st.button("🎯 Find Similar Movies", type="primary", use_container_width=True)

    if search_clicked and movie_input:
        matches = difflib.get_close_matches(movie_input.strip(), all_titles, n=1, cutoff=0.4)
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

            # Searched movie info
            yr  = str(movie_row["year"]) if movie_row["year"] > 0 else "N/A"
            gen = movie_row["genres"].replace(" ", " · ") or "N/A"
            cst = ", ".join(movie_row["cast"].split()[:5]) or "N/A"
            ov  = (movie_row["overview"][:300]+"...") if len(movie_row["overview"])>300 else movie_row["overview"]
            rt  = f"{int(movie_row['runtime'])} min" if movie_row["runtime"] > 0 else "N/A"
            lang = movie_row["original_language"].upper()
            ctry = movie_row["countries_str"] or "N/A"
            st.markdown(f"""
            <div class='searched-card'>
              <div style='color:#fff;font-size:1.3em;font-weight:700;margin-bottom:6px'>🎬 {matched_title}</div>
              <div style='color:#a78bfa;font-size:0.85em;margin-bottom:8px'>
                {yr} &nbsp;·&nbsp; {rt} &nbsp;·&nbsp; ⭐ {movie_row['vote_average']:.1f}/10
                &nbsp;·&nbsp; {int(movie_row['vote_count']):,} votes
                &nbsp;·&nbsp; Lang: {lang} &nbsp;·&nbsp; {ctry}
              </div>
              <div style='color:#7c3aed;font-size:0.82em;margin-bottom:8px'>{gen}</div>
              <div style='color:#94a3b8;font-size:0.8em;margin-bottom:4px'><b>Cast:</b> {cst}</div>
              <div style='color:#cbd5e1;font-size:0.85em;line-height:1.55'>{ov}</div>
            </div>""", unsafe_allow_html=True)

            # Bar chart
            chart_items = [{"title": row["title"], "score": round(s*100,1),
                            "rating": row["vote_average"]} for row,s in results[:10]]
            bar_chart(chart_items, "#7c3aed", "#a78bfa", "title", "score",
                      lambda r: f"{r['score']}% &nbsp; ★{r['rating']:.1f}",
                      "📊 Match Score — Top 10 Recommendations")

            # Cards
            st.markdown(f"<div style='color:#a78bfa;font-size:1.1em;font-weight:700;margin:16px 0 10px'>🎬 {len(results)} Similar Movies</div>",
                        unsafe_allow_html=True)
            for i, (row, sim_score) in enumerate(results):
                pct = round(sim_score*100,1)
                w   = max(2, int(pct*5))
                yr2  = str(row["year"]) if row["year"] > 0 else "N/A"
                rt2  = f"{row['runtime']} min" if row["runtime"] > 0 else "N/A"
                gen2 = row["genres"].replace(" "," · ") or "N/A"
                cst2 = ", ".join(row["cast"].split()[:4]) or "N/A"
                ov2  = (row["overview"][:180]+"...") if len(row["overview"])>180 else row["overview"]
                lang2 = row["original_language"].upper()
                ctry2 = row["countries_str"] or "N/A"
                st.markdown(f"""
                <div class='movie-card'>
                  <div class='movie-title'>{i+1}. {row['title']}</div>
                  <div class='movie-meta'>{yr2} &nbsp;·&nbsp; {rt2} &nbsp;·&nbsp;
                    ⭐ {row['vote_average']:.1f}/10 &nbsp;·&nbsp; Lang: {lang2} &nbsp;·&nbsp; {ctry2}</div>
                  <div style='color:#7c3aed;font-size:0.78em;margin-bottom:6px'>{gen2}</div>
                  <div style='margin:6px 0 8px'>
                    <div style='color:#7c3aed;font-size:0.75em;margin-bottom:3px'>Match: {pct}%</div>
                    <div class='bar-bg'><div class='bar-fill' style='width:{w}%'></div></div>
                  </div>
                  <div style='color:#94a3b8;font-size:0.78em;margin-bottom:4px'><b>Cast:</b> {cst2}</div>
                  <div class='movie-overview'>{ov2}</div>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — Top Rated
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🏆 Top Rated Movies")
    c1,c2,c3 = st.columns(3)
    with c1: genre_top   = st.selectbox("Genre",    genre_list,            key="gtop")
    with c2: lang_top    = st.selectbox("Language", list(LANG_MAP.keys()), key="ltop")
    with c3: country_top = st.selectbox("Country",  country_list,          key="cotop")
    count_top = st.slider("How many to show", 5, 50, 20, 1, key="ctop")

    if st.button("▶ Load Top Rated", type="primary", use_container_width=True, key="btop"):
        filtered = df[df["vote_count"] >= 50].copy()
        filtered = filter_by_lang_country(filtered, lang_top, country_top)
        if genre_top != "All":
            filtered = filtered[filtered["genres"].str.contains(genre_top, case=False, na=False)]

        if filtered.empty:
            st.warning("No movies found for this combination. Try different filters.")
        else:
            top = filtered.nlargest(int(count_top), "weighted_score")
            items = [{"title": r["title"], "score": r["weighted_score"],
                      "rating": r["vote_average"], "votes": r["vote_count"]}
                     for _, r in top.iterrows()]
            bar_chart(items, "#059669", "#34d399", "title", "score",
                      lambda r: f"★{r['rating']:.1f} ({int(r['votes']):,} votes)",
                      "📊 Weighted Rating Chart")
            for i, (_, r) in enumerate(top.iterrows()):
                lang = r["original_language"].upper()
                ctry = r["countries_str"] or "N/A"
                movie_card(i+1, r["title"], r["year"], r["runtime"],
                           r["vote_average"], r["vote_count"], r["genres"],
                           r["cast"], r["overview"],
                           extra=f"&nbsp;·&nbsp; Dir: {r['director'] or 'N/A'} &nbsp;·&nbsp; Lang: {lang} &nbsp;·&nbsp; {ctry}")

# ══════════════════════════════════════════════════════════════
# TAB 3 — Most Popular
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔥 Most Popular Movies")
    c4,c5,c6 = st.columns(3)
    with c4: genre_pop   = st.selectbox("Genre",    genre_list,            key="gpop")
    with c5: lang_pop    = st.selectbox("Language", list(LANG_MAP.keys()), key="lpop")
    with c6: country_pop = st.selectbox("Country",  country_list,          key="copop")
    count_pop = st.slider("How many to show", 5, 50, 20, 1, key="cpop")

    if st.button("▶ Load Popular Movies", type="primary", use_container_width=True, key="bpop"):
        filtered2 = df.copy()
        filtered2 = filter_by_lang_country(filtered2, lang_pop, country_pop)
        if genre_pop != "All":
            filtered2 = filtered2[filtered2["genres"].str.contains(genre_pop, case=False, na=False)]

        if filtered2.empty:
            st.warning("No movies found for this combination. Try different filters.")
        else:
            top2 = filtered2.nlargest(int(count_pop), "popularity")
            items2 = [{"title": r["title"], "score": r["popularity"],
                       "rating": r["vote_average"]}
                      for _, r in top2.iterrows()]
            bar_chart(items2, "#dc2626", "#f87171", "title", "score",
                      lambda r: f"{r['score']:.0f} pts &nbsp; ★{r['rating']:.1f}",
                      "📊 Popularity Score Chart")
            for i, (_, r) in enumerate(top2.iterrows()):
                lang = r["original_language"].upper()
                ctry = r["countries_str"] or "N/A"
                movie_card(i+1, r["title"], r["year"], r["runtime"],
                           r["vote_average"], r["vote_count"], r["genres"],
                           r["cast"], r["overview"],
                           extra=f"&nbsp;·&nbsp; Popularity: {r['popularity']:.0f} &nbsp;·&nbsp; Lang: {lang} &nbsp;·&nbsp; {ctry}")

# ══════════════════════════════════════════════════════════════
# TAB 4 — About
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"""
## 🎬 About Movie Recommender

**Movie Recommender** helps you instantly find films similar to the ones you love —
no account, no sign-up, works on any device in any language.

---
### 🧠 How recommendations work
| Step | What happens |
|------|-------------|
| 1 | Each movie's genres, keywords, cast, director, tagline & overview are combined into one text |
| 2 | **TF-IDF** converts that text into a numerical vector |
| 3 | **Cosine Similarity** finds the closest matching movies |
| 4 | **Fuzzy matching** handles typos |

### 🌍 Language & Country filters
Filter any browse tab by language (Hindi, English, French, Korean etc.) and by production country.

### 📂 Dataset
**{len(df):,} movies** from the TMDB 5000 Movie Dataset

### 🛠️ Built with
Python · Scikit-learn · Pandas · Streamlit
    """)
