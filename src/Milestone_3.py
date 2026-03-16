#!/usr/bin/env python
# coding: utf-8

# ## 1. Load the Feature Engineered Data

# In[11]:


import pandas as pd

df = pd.read_csv("../data/processed/netflix_feature_engineered.csv")

print(df.shape)
df.head()


# In[12]:


df.columns


# ## 2. Select Modeling Features

# In[13]:


df['duration_int'] = df['duration'].str.extract('(\d+)')
df['duration_int'] = df['duration_int'].astype(float)
df['primary_country'] = df['country'].str.split(',').str[0]
df['primary_country'] = df['primary_country'].fillna("Unknown")
df['main_genre'] = df['listed_in'].str.split(',').str[0]
def length_category(x):
    if x < 60:
        return "Short"
    elif x < 120:
        return "Medium"
    else:
        return "Long"

df['content_length_category'] = df['duration_int'].apply(length_category)
df['is_original'] = df['description'].str.contains(
    "Netflix", case=False, na=False
)
df['is_original'] = df['is_original'].astype(int)


# In[14]:


df.columns


# ## 3. Convert Categorical → Numerical

# In[15]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['type_encoded'] = le.fit_transform(df['type'])
df['rating_encoded'] = le.fit_transform(df['rating'])
df['country_encoded'] = le.fit_transform(df['primary_country'])
df['genre_encoded'] = le.fit_transform(df['main_genre'])
df['length_encoded'] = le.fit_transform(df['content_length_category'])


# ## 4. Select Final Modeling Dataset

# In[16]:


model_df = df[[
    'release_year',
    'duration_int',
    'rating_encoded',
    'country_encoded',
    'genre_encoded',
    'length_encoded',
    'type_encoded'
]]

model_df.head()


# ## Save Modeling Dataset

# In[17]:


model_df.to_csv("../data/processed/netflix_modeling_data.csv", index=False)


# # Clustering Analysis
# 

# ### 1. Load Data
# 

# In[18]:


df = pd.read_csv("../data/processed/netflix_modeling_data.csv")


# ### 2. Select Clustering Features

# In[19]:


X = df[['duration_int','rating_encoded','genre_encoded']]


# ### 3. Standardize Data

# In[20]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ### 4. Find Optimal Clusters (Elbow Method)

# In[22]:


model_df.isnull().sum()


# In[23]:


model_df = model_df.dropna()


# In[24]:


X = model_df


# In[25]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[26]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []

for k in range(1,10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,10), inertia)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()


# ### 5. Train Clustering Model

# In[28]:


kmeans = KMeans(n_clusters=4, random_state=42)

model_df['cluster'] = kmeans.fit_predict(X_scaled)

model_df.head()


# ### 6. Cluster Insights

# In[30]:


cluster_summary = model_df.groupby('cluster').mean()

print(cluster_summary)


# # Classification Model
# 

# ### 1. Prepare Data

# In[31]:


from sklearn.model_selection import train_test_split

X = df.drop(columns=['type_encoded'])
y = df['type_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ### 2. Train Model (Random Forest)

# In[32]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)


# ### 3. Predictions

# In[33]:


y_pred = model.predict(X_test)


# ### 4. Evaluation

# In[34]:


from sklearn.metrics import classification_report, accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # Advanced Analytics (Feature Importance)

# ### 1. Feature Importance

# In[35]:


import pandas as pd

importance = model.feature_importances_

features = X.columns

importance_df = pd.DataFrame({
    'feature': features,
    'importance': importance
})

importance_df = importance_df.sort_values(
    by='importance',
    ascending=False
)

print(importance_df)


# ### 2. Plot Feature Importance

# In[36]:


import seaborn as sns

sns.barplot(
    data=importance_df,
    x='importance',
    y='feature'
)

plt.title("Feature Importance for Content Type")
plt.show()


# ### 3.Correlation Analysis

# In[37]:


plt.figure(figsize=(8,6))

sns.heatmap(
    model_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Matrix")
plt.show()


# ### 4. Country vs Genre Analysis

# In[43]:


df.head()
df.columns
df.info()


# In[42]:


country_genre = pd.crosstab(
    df['country_encoded'],
    df['genre_encoded']
)

country_genre.head()


# In[44]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(country_genre.head(10), cmap="coolwarm")

plt.title("Country vs Genre Distribution")
plt.show()


# #### 5. PCA Visualization (Advanced Machine Learning)

# In[48]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
model_df = model_df.dropna()
X = model_df.drop(columns=['cluster'], errors='ignore')


# In[52]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
import matplotlib.pyplot as plt

plt.scatter(X_pca[:,0], X_pca[:,1], c=model_df['cluster'])

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Netflix Content Clusters")

plt.show()


# ### 6.Regression Analysis (Extra Advanced)

# In[54]:


from sklearn.linear_model import LinearRegression

X_reg = model_df.drop(columns=['duration_int'])
y_reg = model_df['duration_int']

model_reg = LinearRegression()

model_reg.fit(X_reg, y_reg)

pred = model_reg.predict(X_reg)


# In[55]:


from sklearn.metrics import r2_score

print("R2 Score:", r2_score(y_reg, pred))


# In[ ]:




