
# coding: utf-8

# In[1]:


# Importing libraries
import nltk
from nltk import FreqDist
import pandas as pd
import numpy as np
import re
import gensim
from gensim import corpora
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import seaborn as sns
from nltk.corpus import wordnet
import re
from numpy import array
import numpy as np
import pandas as pd
import pyLDAvis.gensim
nltk.download('averaged_perceptron_tagger')
  


# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/manan-arya/ChatBot/master/reviews2.csv')


# In[2]:


# findinf frequence of words
def freq_words(x, terms = 30):
    all_words = ' '.join([str(text) for text in x])
    all_words = all_words.split()

    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()


# In[3]:


# category = pd.read_csv('category_file.csv')
# for i in range(len(category['Words'])):
#     category['Words'][i]=''.join([j for j in category['Words'][i] if j.isalpha() or j==' '])


# In[7]:


# function to remove stopwords
def remove_stopwords(rev):
    synonyms = [] 
    antonyms = [] 

#     for syn in wordnet.synsets("easy"): 
#       for l in syn.lemmas(): 
#           synonyms.append(l.name()) 
#           if l.antonyms(): 
#              antonyms.append(l.antonyms()[0].name()) 

    #print(set(synonyms)) 
    #print(set(antonyms))

    stop_words = stopwords.words('english') + synonyms + antonyms
    more_stopwords = pd.read_csv('more_stopwords.csv')
    for i in more_stopwords.values:
        if i not in stop_words:
            stop_words.append(i)
    #   stop_words.append(re.findall(r'\w+', category['Words'][])
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new


# In[5]:


# user porterstemmer for stemming
def stemming(texts): 
    s_stemmer = PorterStemmer()
    for i in range(len(texts)):
#         texts[i]=[s_stemmer.stem(texts[i][j]) for j in range(len(texts[i]))]
        texts[i]=s_stemmer.stem(texts[i])
    return texts


# In[6]:


# Read the csv file
df = pd.read_csv('reviews2.csv')


# replace "n't" with " not"
df['Review'] = df['Review'].str.replace("n\'t", " not")

# remove unwanted characters, numbers and symbols
df['Review'] = df['Review'].str.replace("[^a-zA-Z#]", " ")

# remove short words (length < 3)
df['Review'] = df['Review'].apply(lambda x: ' '.join([str(w) for w in str(x).split() if len(w)>2]))


# In[ ]:


# remove stopwords from the text
reviews = [remove_stopwords(r.split()) for r in df['Review']]

# # stem all the words
reviews = [stemming(r.split()) for r in reviews]
reviews = ([' '.join(i) for i in reviews])
reviews

# make entire text lowercase/
reviews = [r.lower() for r in reviews]
        
reviews = pd.Series(reviews).apply(lambda x: x.split())
# tokenized_reviews
# reviews_2 = stemming(tokenized_reviews)


# In[102]:


# tokenized_reviews


# In[99]:


nltk.download('words')


# In[100]:


tagged = []
for i in range(len(reviews)):
  tagged.append(nltk.pos_tag(reviews[i]))
tagged
#namedEnt = nltk.ne_chunk(tagged, binary=True)
#namedEnt


# In[103]:


reviews_2 = []
noun_list = ['NN','NNS','NNP','NNPS']
for i in range(len(tagged)):
  temp = []
  for j in range(len(tagged[i])):
    if tagged[i][j][1] in noun_list:
      temp.append(tagged[i][j][0])
  reviews_2.append(temp)
print(reviews_2)


# In[104]:


# Create the term dictionary of our corpus, where every unique term is assigned an index
dictionary = corpora.Dictionary(reviews_2)#reviews_2)

# Convert list of reviews (reviews_2) into a Document Term Matrix using the dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]#reviews_2]
   


# In[111]:


reviews# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix,
                id2word=dictionary,
                num_topics=7, 
                random_state=100,
                chunksize=1000,
                passes=50)

lda_model.print_topics()

# Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
# vis

pyLDAvis.enable_notebook()
lda_disp = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary, sort_topics=False)
pyLDAvis.display(lda_disp)

# Print the Keyword in the 10 topics
#lda_model.print_topics() 


# In[81]:


lda_model.print_topics()


# In[82]:


# Print the Keyword in the 10 topics
classes = lda_model.print_topics() 
classes_names = []
lda_out = []
cat = []
for i in range(len(classes)):
  cat.append(classes[i][0])
  classes[i] = list(classes[i])
  classes[i][1] = re.split('"|,|\'|',classes[i][1])
  classes[i][1]
  words = []
  for word in classes[i][1]:
    if word.isalpha():
      words.append(word)
  lda_out.append(words)
  
dict_words = {'Category':cat,'Words':lda_out}
lda_out = pd.DataFrame(dict_words)
lda_out


# In[83]:


category_file = lda_out.to_csv('category_file.csv',index = False).

