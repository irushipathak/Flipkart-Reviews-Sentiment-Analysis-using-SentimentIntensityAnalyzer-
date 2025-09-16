#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis using NLP

#              Sentiment Analysis is the process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the writer's attitude towards a particular topic, product, etc. is positive, negative, or neutral.                  
#              Sentiment Analysis is used to help businesses monitor brand and product sentiment in customer feedback, and understand customer needs.

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt,seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer

pd.set_option('max_colwidth', 100)


# In[28]:


df=pd.read_csv(r'D:\projects\Dataset.csv',encoding='ISO-8859-1')


# In[5]:


df.head()


# In[6]:


print(df.shape)


# In[7]:


df=df.head(1000)


# In[8]:


df.shape


# In[9]:


df.Rate.value_counts()


# In[10]:


df.Rate.value_counts().sort_index().plot(kind='bar',title='Ratings',xlabel='Stars',ylabel='Counts')


# In[11]:


test = df['Summary'][500]


# In[12]:


tokens=nltk.word_tokenize(test)


# In[13]:


tokens


# In[14]:


tags=nltk.pos_tag(tokens)
tags[:10]


# In[15]:


chunks = nltk.chunk.ne_chunk(tags)
chunks.pprint()


# In[16]:


from nltk.sentiment import SentimentIntensityAnalyzer 


# In[17]:


sia = SentimentIntensityAnalyzer()


# In[18]:


print(test)
sia.polarity_scores(test)


# In[19]:


result=df.Summary.apply(lambda x: sia.polarity_scores(x))


# In[20]:


resultls=list(result)


# In[21]:


scores=pd.DataFrame(resultls)


# In[22]:


merged_scores=scores.merge(df,left_index=True,right_index=True)


# In[23]:


merged_scores


# In[24]:


sns.barplot(data=merged_scores,x='Rate',y='compound')


# In[25]:


sns.barplot(data=merged_scores,x='Rate',y='pos')


# In[26]:


sns.barplot(data=merged_scores,x='Rate',y='neg')


# In[27]:


fig , axis= plt.subplots(1,3,figsize=(15,3))
sns.barplot(data=merged_scores,x='Rate',y='pos',ax=axis[0])
sns.barplot(data=merged_scores,x='Rate',y='neg',ax=axis[1])
sns.barplot(data=merged_scores,x='Rate',y='neu',ax=axis[2])
axis[0].set_title('Positive')
axis[1].set_title('Negative')
axis[2].set_title('Neutral')
plt.tight_layout()
plt.show()


# In[ ]:




