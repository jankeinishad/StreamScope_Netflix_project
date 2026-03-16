

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="StreamScope Dashboard", layout="wide")

st.title("🎬 StreamScope: Netflix Content Strategy Dashboard")

# Load dataset
df = pd.read_csv("data/processed/netflix_cleaned_featured.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Sidebar Filters
st.sidebar.header("Filters")

# Year filter
if 'release_year' in df.columns:
    year_range = st.sidebar.slider(
        "Release Year",
        int(df['release_year'].min()),
        int(df['release_year'].max()),
        (2010, 2020)
    )
    df = df[(df['release_year'] >= year_range[0]) & (df['release_year'] <= year_range[1])]

# Content type filter
if 'type' in df.columns:
    type_filter = st.sidebar.multiselect(
        "Content Type",
        df['type'].unique(),
        default=df['type'].unique()
    )
    df = df[df['type'].isin(type_filter)]

# Rating filter
if 'rating' in df.columns:
    rating_filter = st.sidebar.multiselect(
        "Rating",
        df['rating'].unique(),
        default=df['rating'].unique()
    )
    df = df[df['rating'].isin(rating_filter)]

st.subheader("Filtered Dataset Size")
st.write(df.shape)

# -------------------------
# Netflix Growth Over Time
# -------------------------

if 'release_year' in df.columns:
    st.subheader("📈 Netflix Content Growth")

    growth = df['release_year'].value_counts().sort_index()

    fig1 = px.line(
        x=growth.index,
        y=growth.values,
        labels={"x": "Year", "y": "Number of Titles"},
        title="Netflix Content Growth Over Time"
    )

    st.plotly_chart(fig1, use_container_width=True)

# -------------------------
# Movie vs TV Shows
# -------------------------

if 'type' in df.columns:
    st.subheader("🎥 Movies vs TV Shows")

    type_counts = df['type'].value_counts()

    fig2 = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title="Content Type Distribution"
    )

    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Rating Distribution
# -------------------------

if 'rating' in df.columns:
    st.subheader("⭐ Rating Distribution")

    fig3 = px.histogram(
        df,
        x="rating",
        title="Distribution of Ratings"
    )

    st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Top Countries
# -------------------------

if 'primary_country' in df.columns:
    st.subheader("🌍 Top Content Producing Countries")

    country_counts = df['primary_country'].value_counts().head(10)

    fig4 = px.bar(
        x=country_counts.index,
        y=country_counts.values,
        labels={"x": "Country", "y": "Number of Titles"},
        title="Top Countries on Netflix"
    )

    st.plotly_chart(fig4, use_container_width=True)

# -------------------------
# Country Map Visualization
# -------------------------

if 'primary_country' in df.columns:
    st.subheader("🌎 Global Netflix Content Map")

    country_map = df['primary_country'].value_counts().reset_index()
    country_map.columns = ['country', 'count']

    fig5 = px.choropleth(
        country_map,
        locations="country",
        locationmode="country names",
        color="count",
        title="Netflix Content Distribution by Country"
    )

    st.plotly_chart(fig5, use_container_width=True)

# -------------------------
# Genre Analysis
# -------------------------

if 'listed_in' in df.columns:
    st.subheader("🎭 Top Genres")

    genre_counts = df['listed_in'].value_counts().head(10)

    fig6 = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        labels={"x": "Genre", "y": "Number of Titles"},
        title="Top Genres on Netflix"
    )

    st.plotly_chart(fig6, use_container_width=True)

# -------------------------
# Animated Growth Chart
# -------------------------

if 'release_year' in df.columns and 'type' in df.columns:
    st.subheader("🎬 Animated Netflix Growth")

    yearly_data = df.groupby(['release_year', 'type']).size().reset_index(name='count')

    fig7 = px.bar(
        yearly_data,
        x="type",
        y="count",
        animation_frame="release_year",
        color="type",
        title="Netflix Content Growth Animation"
    )

    st.plotly_chart(fig7, use_container_width=True)

# -------------------------
# Key Insights
# -------------------------

st.subheader("📊 Key Insights")

st.write("""
- Netflix experienced rapid growth in content production after 2015.
- Movies dominate the catalog compared to TV Shows.
- The United States and India contribute a significant number of titles.
- Netflix increasingly focuses on international content to attract global audiences.
- TV-MA and TV-14 ratings are the most common across the platform.
""")

st.markdown("---")
st.markdown("StreamScope Project | Data Analytics Internship")