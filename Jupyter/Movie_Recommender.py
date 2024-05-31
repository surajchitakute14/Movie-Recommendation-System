#!/usr/bin/env python
# coding: utf-8

# In[119]:


import numpy as np
import pandas as pd


# In[120]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[121]:


movies.head(1)


# In[122]:


credits.head(1)


# In[123]:


movies = movies.merge(credits,on='title')


# In[124]:


movies.head(1)


# In[125]:


# genres
# id
# keywords
# title
# overview
# cast
# crew

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[126]:


movies.head()


# In[127]:


movies.isnull().sum()


# In[128]:


movies.dropna(inplace = True)


# In[129]:


movies.duplicated().sum()


# In[130]:


movies.iloc[0].genres


# In[131]:


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'


# In[132]:


import ast


# In[133]:



def convert(obj):
    L = []
    for i in ast.literal_eval(obj) :
        L.append(i['name'])
    return L


# In[134]:


movies['genres'] = movies['genres'].apply(convert)


# In[135]:


movies.head()


# In[136]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[137]:


movies['cast'][0]


# In[138]:



def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj) :
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[139]:


movies['cast'] = movies['cast'].apply(convert3)


# In[140]:


movies.head()


# In[141]:


movies['crew'][0]


# In[142]:



def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj) :
        if i['job'] == 'Director' :
            L.append(i['name'])
            break
    return L


# In[143]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[144]:


movies.head()


# In[145]:


movies['overview'][0]


# In[146]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[147]:


movies.head()


# In[148]:


movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[149]:


movies.head()


# In[150]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[151]:


movies.head()


# In[152]:


new_df = movies[['movie_id','title','tags']]


# In[153]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[154]:


new_df.head()


# In[155]:


new_df['tags'][0]


# In[156]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[157]:


new_df.head()


# In[158]:


new_df['tags'][0]


# In[159]:


new_df['tags'][1]


# In[182]:


import nltk


# In[183]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[184]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[185]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[ ]:





# In[186]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[187]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[188]:


vectors


# In[189]:


vectors[0]


# In[190]:


cv.get_feature_names() 


# In[191]:


vectors.shape


# In[192]:


from sklearn.metrics.pairwise import cosine_similarity


# In[194]:


similarity = cosine_similarity(vectors)


# In[201]:


sorted(list(enumerate(similarity[0])),reverse = True , key = lambda x : x[1])[1:6]


# In[212]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse = True , key = lambda x : x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[214]:


recommend('Batman Begins')


# In[205]:


new_df.iloc[1216].title


# In[215]:


import pickle


# In[216]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[218]:


new_df['title'].values


# In[220]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[221]:


pickle.dump(similarity,open("similarity.pkl",'wb'))


# In[ ]:




