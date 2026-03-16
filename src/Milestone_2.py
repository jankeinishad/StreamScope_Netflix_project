#!/usr/bin/env python
# coding: utf-8

# ## MILESTONE 2
# 

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# In[7]:


df = pd.read_csv("../data/processed/netflix_cleaned_featured.csv")


# In[8]:


print(df.shape)
df.head()


# ## Netflix Growth Over Time

# In[9]:


plt.figure(figsize=(10,6))
df.groupby('year_added').size().plot(kind='line', marker='o')
plt.title("Netflix Content Growth Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Titles Added")
plt.show()


# ## Content Type Distribution

# In[10]:


plt.figure(figsize=(6,4))
sns.countplot(data=df, x='type')
plt.title("Movies vs TV Shows Distribution")
plt.show()


# ## Content Type Pie Chart

# In[11]:


df['type'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6,6))
plt.title("Content Type Share")
plt.ylabel("")
plt.show()


# ## Top 10 Genres

# In[12]:


from collections import Counter

all_genres = df['listed_in'].str.split(',').explode()
top_genres = all_genres.value_counts().head(10)

plt.figure(figsize=(8,5))
sns.barplot(x=top_genres.values, y=top_genres.index)
plt.title("Top 10 Genres")
plt.show()


# ## Top 10 Countries

# In[13]:


top_countries = df['primary_country'].value_counts().head(10)

plt.figure(figsize=(8,5))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title("Top 10 Countries by Content")
plt.show()


# ## Rating Distribution (Histogram)

# In[14]:


plt.figure(figsize=(8,5))
sns.histplot(df['rating'], bins=15)
plt.title("Rating Distribution")
plt.xticks(rotation=45)
plt.show()


# ## Duration Distribution

# In[15]:


plt.figure(figsize=(8,5))
sns.histplot(df[df['type']=="Movie"]['duration_num'], bins=30)
plt.title("Movie Duration Distribution")
plt.show()


# ## Content Age Distribution

# In[16]:


plt.figure(figsize=(8,5))
sns.histplot(df['content_age'], bins=30)
plt.title("Content Age Distribution")
plt.show()


# ## Hypothesis Testing

# ### Hypothesis 1: Duration Difference (T-Test)

# In[17]:


from scipy.stats import ttest_ind

movies = df[df['type']=="Movie"]['duration_num'].dropna()
shows = df[df['type']=="TV Show"]['duration_num'].dropna()

t_stat, p_value = ttest_ind(movies, shows)

print("T-statistic:", t_stat)
print("P-value:", p_value)


# In[18]:


from scipy.stats import ttest_ind

movies = df[df['type']=="Movie"]['duration_num'].dropna()
shows = df[df['type']=="TV Show"]['duration_num'].dropna()

t_stat, p_value = ttest_ind(movies, shows)

print("T-statistic:", t_stat)
print("P-value:", p_value)


# ### Hypothesis 2: Country vs Original (Chi-Square)

# In[19]:


from scipy.stats import chi2_contingency

contingency = pd.crosstab(df['primary_country'], df['is_original'])
chi2, p, dof, expected = chi2_contingency(contingency)

print("Chi-square:", chi2)
print("P-value:", p)


# ## Correlation Heatmap

# In[20]:


numeric_cols = ['duration_num', 'release_year', 'year_added', 'content_age', 'genre_count']

plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[21]:


import pandas as pd
df = pd.read_csv("../data/raw/netflix_titles.csv")


# In[22]:


df.to_csv("../data/processed/netflix_feature_engineered.csv", index=False)


# In[ ]:




