#!/usr/bin/env python
# coding: utf-8

# ## Introduction to the COVID-19 Open Research Dataset
# 
# The COVID-19 Open Research Dataset (CORD-19) is a collection of over 50,000 scholarly articles - including over 40,000 with full text - about COVID-19, SARS-CoV-2, and related coronaviruses. This dataset has been made freely available with the goal to aid research communities combat the COVID-19 pandemic. It has been made available by the Allen Institute for AI in partnership with leading research groups to prepare and distribute the COVID-19 Open Research Dataset (CORD-19), in response to the COVID-19 pandemic.
# 
# During this lab you will learn how to process and analyze a subset of the articles present in the dataset, group them together into a series of clusters, and use Automated ML to train a machine learning model capable of classifying new articles as they are published.

# ### Setup
# 
# We will start off by installing a few packages, such as `nltk` for text processing and `wordcloud`, `seaborn`, and `yellowbrick` for various visualizations.

# In[1]:


get_ipython().run_line_magic('pip', 'install nltk')
get_ipython().run_line_magic('pip', 'install wordcloud')
get_ipython().run_line_magic('pip', 'install seaborn')
get_ipython().run_line_magic('pip', 'install yellowbrick')


# We'll first download stopwords and the Punkt tokenizer models present in the `nltk` package, in order to be able to process the articles

# In[2]:


import nltk

nltk.download('punkt')
nltk.download('stopwords')


# We'll also import the rest of the modules needed in this notebook, and do a quick sanity-check on the Azure ML SDK version

# In[3]:


import os
import json
from string import punctuation

import pandas as pd
import seaborn as sns
sns.set_palette('Set2')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, Birch, AgglomerativeClustering
from sklearn.metrics import roc_auc_score
from nltk import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer, PorterStemmer

from azureml.core import Workspace, Datastore, Dataset, VERSION

print("Azure ML SDK Version: ", VERSION)


# ## Load the Covid-19 data
# 
# CORD-19 has been uploaded to an Azure Storage Account, we will connect to it and use it's API to download the dataset locally. Also please make sure to update the storage account name from the labguide in the below cell. 

# In[4]:


covid_dirname = 'covid19temp'

cord19_dataset = Dataset.File.from_files('https://aiinadaystorage800633.blob.core.windows.net/' + covid_dirname )
mount = cord19_dataset.mount()

covid_dirpath = os.path.join(mount.mount_point, covid_dirname)


# Display a sample of the dataset (top 5 rows).

# In[5]:


mount.start()

# Load metadata.csv, as it contains a list of all the articles and their corresponding titles/authors/contents.
metadata_filename = os.path.join(covid_dirpath, 'metadata.csv')

metadata = pd.read_csv(metadata_filename)
metadata.sample(5)


# Some of the articles do not have any associated documents, so we will filter those out.

# In[6]:


metadata_with_docs = metadata[metadata['pdf_json_files'].isna() == False]

print(f'Dataset contains {metadata.shape[0]} entries, out of which {metadata_with_docs.shape[0]} have associated json documents')


# Display the percentage of items in the dataset that have associated JSON documents (research papers that have extra metadata associated with them).

# In[7]:


doc_counts = [metadata_with_docs.shape[0], metadata.shape[0] - metadata_with_docs.shape[0]]
doc_labels = ['Entries with associated json documents', 'Entries without any associated json documents']

fig, ax = plt.subplots()
ax.pie(doc_counts, labels=doc_labels, autopct='%1.1f%%')
ax.axis('equal')
plt.show()


# ## Investigate individual items
# 
# Let's load an example entry from the dataset.

# In[8]:


# Change the document index in order to preview a different article
DOCUMENT_INDEX = 0 
example_entry = metadata_with_docs.iloc[DOCUMENT_INDEX]

filepath = os.path.join(covid_dirpath, example_entry['pdf_json_files'])
print(f'Document local filepath: {filepath}')
print(f'Document local filepath: {covid_dirpath}')

filepath = covid_dirpath + '/comm_use_subset/pdf_json/02a009e42054081b441d0f4b203679c4b0cae38d.json'
print(f'Document local filepath: {filepath}')


# Next, we will display the list of elements that are available for the selected document.

# In[9]:


try:
    with open(filepath, 'r') as f:
        data = json.load(f)
        
except FileNotFoundError as e:
    # in case the mount context has been closed
    mount.start()
    with open(filepath, 'r') as f:
        data = json.load(f)
        
print(f'Data elements: { ", ".join(data.keys())}' )


# Please make sure to update the storage account name from the labguide in the below cell.

# In[10]:


from azureml.core import  Dataset
cord19_dataset = Dataset.File.from_files('https://aiinadaystorage800633.blob.core.windows.net/covid19temp')
mount = cord19_dataset.mount()


# View the full text version of the document.

# In[11]:


for p in data['body_text']:
    print(p['text'], '\n')


# ## Stop words
# 
# Here's a quote from Stanford's NLP team that will provide some context on stop words and their intended usage:
# 
# _"Sometimes, some extremely common words which would appear to be of little value in helping select documents matching a user need are excluded from the vocabulary entirely. These words are called stop words . The general strategy for determining a stop list is to sort the terms by collection frequency (the total number of times each term appears in the document collection), and then to take the most frequent terms, often hand-filtered for their semantic content relative to the domain of the documents being indexed, as a stop list , the members of which are then discarded during indexing."_
# 
# Let's investigate the stop words list that we will use to clean our data. Note that apart from the standard stopwords, we will also remove any punctuation and also any occurrences of *et al.*, as they are often found in academic articles.

# In[12]:


stop_tokens = nltk.corpus.stopwords.words('english') + list(punctuation) + ['et', 'al.']
print(stop_tokens)


# The code below will be used to read the text associated with a series of articles, remove stop words from their text, and reduce) inflected words to their base form (stemming).
# 
# **NOTE**:
# 
# If you are not familiar with Python code, just execute the following cell and continue with the notebook. Understanding the code below is not require for understanding and following the overall flow of the notebook.

# In[13]:


class Reader:
    """Class used to read the files associated with an article"""
    
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
    
    def read_file_to_json(self, filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except FileNotFoundError as e:
            mount.start()
            with open(filepath, 'r') as f:
                data = json.load(f)
                
        return data
    
    def parse_document(self, document_index):
        document = metadata_with_docs.iloc[document_index]
        
        # One article can have multiple associated documents
        words = []
        for filename in document['pdf_json_files'].split('; '):
            filepath = '{0}/{1}'.format(covid_dirpath, filename)
            pdf_json_files = os.listdir(covid_dirpath + '/comm_use_subset/pdf_json')
            filepath = covid_dirpath + '/comm_use_subset/pdf_json/' + pdf_json_files[document_index]
            
            if document_index % 50 == 0:
                print(filepath)
            
            data = self.read_file_to_json(filepath)



           # Split each paragraph into multiple sentences first, in order to improve the performance of the word tokenizer
            text = data['body_text']
            for paragraph in text:
                p_sentences = sent_tokenize(paragraph['text'])



               # Split each sentence into words, while making sure to remove the stopwords and stem the words
                for p_sentence in p_sentences:
                    sentence = [ self.stemmer.stem(word) for word in word_tokenize(p_sentence) if word.isalpha() and word.lower() not in stop_tokens ]
                    words.extend(sentence)
    
        return (words, document['cord_uid'])
        



class Corpus:
    """An iterator that reads all sentences from the first N documents"""
    
    def __init__(self, n_documents):
        self.n_documents = n_documents
        self.stemmer = SnowballStemmer('english')
        self.reader = Reader()
        
    def __iter__(self):
         for document_index in range(0, self.n_documents):   
            words, document_id = self.reader.parse_document(document_index)
            yield TaggedDocument(words, document_id)
            
    def plain_iter(self):
        for document_index in range(0, self.n_documents):  
            words, document_id = self.reader.parse_document(document_index)
            yield (words, document_id)


# ## Encoding documents as vectors
# 
# In this lab, we're using a subset of 1000 articles to train a Machine Learning model that encodes text documents into numerical vectors (a document embedding model). 
# 
# Training a document embedding model takes a significant amount of time, and for this reason we already provide a trained model. We also provide the code below in case you want to get more details about the process. Running the next two cells will result in loading the already existing model.

# In[14]:


N_DOCUMENTS = 500


# In[15]:


get_ipython().run_cell_magic('time', '', "\nmodel_filename = f'covid_embeddings_model_{N_DOCUMENTS}_docs.w2v'\n\nif (os.path.exists(model_filename)):\n    model = Doc2Vec.load(model_filename)\n    print(f'Done, loaded word2vec model with { len(model.wv.vocab) } words.')\nelse:\n    model = Doc2Vec(Corpus(N_DOCUMENTS), vector_size=128, batch_words=10)\n    model.save(model_filename)\n    print(f'Done, trained word2vec model with { len(model.wv.vocab) } words.')\n")


# ## Word frequencies
# 
# Let's analyze the relative frequencies of words in the corpus of articles. We will display a word cloud to provide a visual representation of these relative frequencies.

# In[16]:


word_frequencies = { key: model.wv.vocab[key].count for key in model.wv.vocab }


# In[17]:


cloud = WordCloud(width=1024, height=768).generate_from_frequencies(word_frequencies)
plt.figure(figsize=(16,12))
plt.imshow(cloud, interpolation='antialiased')
plt.axis("off")


# ## Embedding documents
# 
# Below is an example on how we embed (transform from text to numerical vector) one of the documents.

# In[18]:


words, doc_id = Reader().parse_document(DOCUMENT_INDEX)
model.infer_vector(words)


# And this is an example of a trivial "document" (containing a single, trivial sentence) going through the same process. Notice how, regardless of the length of the sentence, the result vector is always the same size - the `vector_size` argument used while training the `Doc2Vec` model.
# 
# This is very important in the following stages of the process, when we are working with multiple documents.

# In[19]:


model.infer_vector(['human', 'love', 'cat', 'dog'])


# The resulting vectors will look more or less similar, depending on how different the contents of the articles are themselves. See below the differences resulting from a single word change - some of the values significantly overlap, while others are quite different if not opposite.

# In[20]:


adult_vector = model.infer_vector(['adult', 'love', 'cat', 'dog'])
child_vector = model.infer_vector(['child', 'love', 'cat', 'dog'])
labels = range(0, 128) 

plt.bar(labels, adult_vector, align='center', alpha=0.5)
plt.bar(labels, child_vector, align='center', alpha=0.5)
plt.legend(['Adults', 'Children'])
plt.show()


# Let's now do the same for the same for all the documents we're focusing on.

# In[21]:


get_ipython().run_cell_magic('time', '', "\nword_vectors = []\nids = []\n\nfor (words, doc_id) in Corpus(N_DOCUMENTS).plain_iter():\n    ids.append(doc_id)\n    word_vector = model.infer_vector(words)\n    word_vectors.append(word_vector)\n    if len(word_vectors) % 100 == 0:\n        print(f'Processed {len(word_vectors)} documents.')\n")


# Now that we've finished reading the articles, we can dismount the dataset in order to free up resources

# In[22]:


mount.stop()


# ## Covid-19 documents prepared for Machine Learning
# 
# We'll create a new DataFrame using the word vectors we've just calculated, this is the numerical form of the documents which is ready for Machine Learning workloads.

# In[23]:


wv_df = pd.DataFrame(word_vectors, index=ids)
wv_df


# We'll join the DataFrame containing the numerical embeddings with the original dataset.

# In[24]:


indexed_metadata = metadata_with_docs.set_index('cord_uid')
metadata_with_embeddings = pd.concat([indexed_metadata.iloc[:N_DOCUMENTS], wv_df], axis=1)
metadata_with_embeddings


# ## Preparing for clustering documents
# 
# One of the challenges with clustering is to find the ideal number of clusters to look for. The elbow method is one of the most common approaches.
# 
# We're visualizing an elbow metric (the "distortion" score) and trying to find a point where it stops decreasing with the number of clusters.
# 

# In[25]:


visualizer = KElbowVisualizer(KMeans(), k=(3,20))
visualizer.fit(wv_df)

visualizer.show()


# ## Clustering documents
# 
# We've determined the acceptable value for the clusters, so let's use Machine Learning to determine those clusters. We'll use the classic KMeans algorithm to do this.

# In[26]:


clusterer = KMeans(12 if visualizer.elbow_value_ > 12 else visualizer.elbow_value_)
clusterer.fit(wv_df)
clusters = clusterer.labels_


# We'll perform a quick visual check on the clusters. In order to be able to visualize 128 dimensions (which is the size of the word vectors) in a 2-D space, we'll use the PCA (Principal Component Analysis) dimensionality reduction technique. This will transform our 128-dimensional vectors into 2-dimensional ones that we can display.

# In[27]:


pca = PCA(n_components=2)
pca.fit(wv_df)
result = pca.transform(wv_df)


# Afterwards, we can plot the documents in a simple 2-D chart, and color each one according to their cluster

# In[28]:


sns.set(rc={'figure.figsize':(10, 6), 'figure.facecolor':'white', 'axes.facecolor':'white'})

color_palette = sns.color_palette('Paired')
# Each cluster gets its own color from the palette
cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
plt.scatter(result[:,0], result[:,1], s=50, c=cluster_colors, alpha=0.75)

plt.title(f'{N_DOCUMENTS} documents in {clusterer.n_clusters} clusters')
plt.show()


# We'll add each article's cluster as new column to our combined dataset

# In[29]:


metadata_with_clusters = metadata_with_embeddings
metadata_with_clusters['cluster'] = clusters
metadata_with_clusters


# We can now split our data into two datasets - a **training** one that will be used to train a Machine Learning model, able to determine the cluster that should be assigned to an article, and a **test** one that we'll use to test this classifier.
# 
# We will allocate 80% of the articles to training the Machine Learning model, and the remaining 20% to testing it.

# In[30]:


train, test = train_test_split(metadata_with_clusters, train_size=0.8)
train


# To speed up training, we'll ignore all columns except the word vectors calculated using Doc2Vec. For this reason, we will create a separate dataset just with the vectors.

# In[31]:


columns_to_ignore = ['sha', 'source_x', 'title', 'doi', 'pmcid', 'pubmed_id', 'license', 'abstract', 'publish_time', 'authors', 'journal', 'mag_id',
                     'who_covidence_id', 'arxiv_id', 'pdf_json_files', 'pmc_json_files', 'url', 's2_id' ]
train_data_vectors = train.drop(columns_to_ignore, axis=1)
test_data_vectors = test.drop(columns_to_ignore, axis=1)


# ## Register the training and testing datasets for AutoML availability
# 
# We're registering the training and testing datasets with the Azure Machine Learning datastore to make them available inside Azure Machine Learning Studio and Automated ML.

# In[32]:


# Retrieve your ML workspace
ws = Workspace.from_config()
# Retrieve the workspace's default datastore
datastore = ws.get_default_datastore()


Dataset.Tabular.register_pandas_dataframe(train_data_vectors, datastore, 'COVID19Articles_Train_Vectors')
Dataset.Tabular.register_pandas_dataframe(test_data_vectors, datastore, 'COVID19Articles_Test_Vectors')


# ## Open Azure Machine Learning Studio
# 
# Return to the GitHub repo and follow the instructions from there. You will use Automated ML in Azure Machine Learning Studio to train a classification model that predicts the document cluster for new research articles.
