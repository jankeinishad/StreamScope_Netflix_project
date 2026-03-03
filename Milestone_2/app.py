import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📺 StreamScope - Netflix Content Analyzer")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/processed/netflix_cleaned_featured.csv")

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔎 Filter Options")

# Year Range
year_range = st.sidebar.slider(
    "Select Year Range",
    int(df['year_added'].min()),
    int(df['year_added'].max()),
    (2015, int(df['year_added'].max()))
)

# Content Type
content_type = st.sidebar.multiselect(
    "Select Content Type",
    df['type'].unique(),
    default=df['type'].unique()
)

# Country
country = st.sidebar.multiselect(
    "Select Country",
    sorted(df['primary_country'].dropna().astype(str).unique()),
    default=sorted(df['primary_country'].dropna().astype(str).unique())[:5]
)

# Rating
rating = st.sidebar.multiselect(
    "Select Rating",
    df['rating'].unique(),
    default=df['rating'].unique()
)

# Genre
all_genres = sorted(set(df['listed_in'].str.split(',').explode().str.strip()))

genre = st.sidebar.multiselect(
    "Select Genre",
    all_genres,
    default=all_genres[:5]
)

# Netflix Original Filter
original_filter = st.sidebar.selectbox(
    "Netflix Original?",
    ["All", "Yes", "No"]
)

# -----------------------------
# APPLY FILTERS
# -----------------------------
filtered = df[
    (df['year_added'] >= year_range[0]) &
    (df['year_added'] <= year_range[1]) &
    (df['type'].isin(content_type)) &
    (df['primary_country'].isin(country)) &
    (df['rating'].isin(rating))
]

# Genre filtering
filtered = filtered[
    filtered['listed_in'].apply(lambda x: any(g in x for g in genre))
]

# Original filtering
if original_filter == "Yes":
    filtered = filtered[filtered['is_original'] == True]
elif original_filter == "No":
    filtered = filtered[filtered['is_original'] == False]

# -----------------------------
# KPI SECTION
# -----------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", len(filtered))
col2.metric("Movies", len(filtered[filtered['type']=="Movie"]))
col3.metric("TV Shows", len(filtered[filtered['type']=="TV Show"]))
col4.metric("Unique Countries", filtered['primary_country'].nunique())

st.markdown("---")

# -----------------------------
# GROWTH OVER TIME
# -----------------------------
st.subheader("📈 Content Growth Over Time")

fig1, ax1 = plt.subplots(figsize=(10,4))
filtered.groupby('year_added').size().plot(kind='line', marker='o', ax=ax1)
ax1.set_xlabel("Year")
ax1.set_ylabel("Number of Titles")
st.pyplot(fig1)

# -----------------------------
# CONTENT TYPE DISTRIBUTION
# -----------------------------
col5, col6 = st.columns(2)

with col5:
    st.subheader("🎬 Content Type Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=filtered, x='type', ax=ax2)
    st.pyplot(fig2)

with col6:
    st.subheader("🌍 Top 10 Countries")
    top_countries = filtered['primary_country'].value_counts().head(10)
    fig3, ax3 = plt.subplots()
    sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax3)
    st.pyplot(fig3)

# -----------------------------
# TOP GENRES
# -----------------------------
st.subheader("🎭 Top 10 Genres")

all_genres_filtered = filtered['listed_in'].str.split(',').explode()
top_genres = all_genres_filtered.value_counts().head(10)

fig4, ax4 = plt.subplots()
sns.barplot(x=top_genres.values, y=top_genres.index, ax=ax4)
st.pyplot(fig4)

# -----------------------------
# DURATION DISTRIBUTION
# -----------------------------
st.subheader("⏳ Duration Distribution (Movies)")

movie_data = filtered[filtered['type']=="Movie"]

fig5, ax5 = plt.subplots()
sns.histplot(movie_data['duration_num'], bins=30, ax=ax5)
ax5.set_xlabel("Duration (Minutes)")
st.pyplot(fig5)

# -----------------------------
# CONTENT AGE DISTRIBUTION
# -----------------------------
st.subheader("📅 Content Age Distribution")

fig6, ax6 = plt.subplots()
sns.histplot(filtered['content_age'], bins=30, ax=ax6)
ax6.set_xlabel("Years Between Release & Added")
st.pyplot(fig6)

st.markdown("---")
st.write("✅ Dashboard Updated Based on Selected Filters")