#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# ## Load Dataset

# In[10]:


df = pd.read_csv("../data/raw/netflix_titles.csv")
print(df.shape)
df.head()


# ## Basic Inspection

# In[11]:


df.info()
df.isnull().sum()
df.duplicated().sum()


# ## Remove Duplicates

# In[12]:


df.drop_duplicates(inplace=True)


# ## Handle Missing Values

# In[14]:


df['director'] = df['director'].fillna("Unknown")
df['cast'] = df['cast'].fillna("Not Available")
df['country'] = df['country'].fillna("Unknown")
df['rating'] = df['rating'].fillna(df['rating'].mode()[0])


# ## Convert Date Column

# In[16]:


# Clean whitespace
df['date_added'] = df['date_added'].str.strip()

# Convert to datetime
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Extract features
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month


# In[17]:


df['date_added'].isnull().sum()


# In[18]:


df[df['date_added'].isnull()][['date_added']]


# ## Normalize Country (Primary Country)

# In[19]:


df['primary_country'] = df['country'].apply(lambda x: x.split(',')[0].strip())


# ## Normalize Genres

# In[20]:


df['listed_in'] = df['listed_in'].str.lower()
df['genre_count'] = df['listed_in'].apply(lambda x: len(x.split(',')))


# ## Extract Numeric Duration

# In[21]:


df['duration_num'] = df['duration'].str.extract('(\d+)')
df['duration_num'] = df['duration_num'].astype(float)


# ## Content Length Category

# In[22]:


def categorize_length(row):
    if row['type'] == 'Movie':
        if row['duration_num'] < 60:
            return "Short Movie"
        elif row['duration_num'] <= 120:
            return "Medium Movie"
        else:
            return "Long Movie"
    else:
        if row['duration_num'] == 1:
            return "Limited Series"
        else:
            return "Multi Season"

df['content_length_category'] = df.apply(categorize_length, axis=1)


# ## Release Decade

# In[24]:


df['release_decade'] = (df['release_year'] // 10) * 10


# ## Netflix Original Detection (Basic Logic)

# In[25]:


df['is_original'] = df['title'].str.contains("Netflix", case=False, na=False)


# In[28]:


df.to_csv("../data/processed/netflix_cleaned_featured.csv", index=False)


# In[ ]:




