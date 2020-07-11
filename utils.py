#!/usr/bin/env python
# Author: Sicong Zhao

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import re
import numpy as np
!pip install hdbscan
import hdbscan
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
from sentence_transformers import SentenceTransformer
bert_large_nli = SentenceTransformer('bert-large-nli-stsb-mean-tokens')



def load_data(path):
    """
    Load Dataset from File

    Parameters:
        path (string): Relative data path.
    Return: 
        A Pandas dataframe that contains reviews.
    """
    input_file = pd.read_csv(path)

    return input_file

def stemming(word):
    """
    Stemming a word

    Parameters:
        word (string): The word to be stemmed.
    Return:
        Stemmed word.
    """
    return stemmer.stem(word)

def lemmentize(word):
    """
    Lemmentizing a word

    Parameters:
        word (string): The word to be lemmentized.
    Return:
        Lemmentized word.
    """
    return WordNetLemmatizer().lemmatize(word, pos='v')

def remove_stopwords(sentence, output='list'):
    """
    Remove stopwords in a sentence, and output a list of words or a sentence.

    Parameters:
        sentence (string): The sentence to be processed.
        output   (string): The output format. 'list' or 'string'.
    Return:
        Lemmentized word.
    """
    result=[]
    for token in gensim.utils.simple_preprocess(sentence) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(token)
    if output == 'string':
        result = ' '.join(result)
    return result

def preprocess(s):
    """
    Formatting the review data by lowercase conversion, trimming space, and filter by length.

    Parameters:
        s (string): Sentence to be pre-processed.
    Return: 
        The processed sentence.
    """
    s = s.lower()
    if s == '' or len(s) < 10 or len(s) > 200:
        return None
    else:
        while s[0] == ' ':
            s = s[1:]
        return s

digits   = "([0-9])"
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    """
    Split strings into sentences.

    Parameters:
        text (str): The string to be processed.
    Return:
        Processed string.
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def HDBSCAN(embeding, min_cluster_size, min_samples, alpha):
    """
    HDBSCAN Clustering.

    Parameters:
        embedding           (2D list): The embeddings to be processed.
        min_cluster_size    (int): The minmum number of observations that could form a cluster.
        min_samples         (int): The distance for group splitting.
        alpha   
    Return:
        Cluster object.
    """
    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, \
        min_samples=min_samples, alpha=alpha).fit(embeding)
    return clusters

def AC(embeding, n_clusters, distance):
    """
    Agglomerative Clustering.

    Parameters:
        embedding   (2D list): The embeddings to be processed.
        n_clusters  (int): The number of clusters.
        distance    (float): The distance for group splitting.   
    Return:
        Cluster object.
    """
    clusters = AgglomerativeClustering(n_clusters=n_clusters, \
        distance_threshold=distance).fit(embeding)
    return clusters

def PCA(embeding, n_components):
    x = pd.DataFrame(embeding)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(x)

def normalize(embedding):
    f = np.array(embedding)
    return (f - f.mean(axis=0)) / f.std(axis=0)

def sentence_embedding(review_list):
    embed = bert_large_nli.encode(review_list)
    return embed

def main(df, threshold, thrd, positive=True, components=0, method='HBDSCAN'):
    '''
    Clustering embedded sentences.
    
    Parameters:
        df          (obj):   The dataframe that contains reviews.
        threshold   (float): The threshold for clustering distance.
        thrd        (float): The threshold for deciding if a review is positive or negative.
        positive    (bool):  Decide if clustering positive reviews or negative reviews.
        components  (int):   If components > 0, apply PCA with 'n_components=components'.
        method      (string):Decide which clustering algorithm to use.
    Return: 
        The dataframe with cluster id assigned to each review.
    '''
    # Split by Sentiment
    if positive:
        df_ = df[df['polarity'] > thrd].copy()
    else:
        df_ = df[df['polarity'] < thrd].copy()
    
    embed = sentence_embedding(df_['review'].values)
    # Normalize embedding
    embed = normalize(embed)
    # PCA
    if components > 0:
        embed = PCA(embed, components)

    # Clustering
    if method == 'AC':
        clust = AC(embed, None, threshold)
    elif method == 'HDBSCAN':
        clust = HDBSCAN(embed, 2, 1, 1.3)
    df_['cluster'] = clust.labels_
    return df_.sort_values(by='cluster').reset_index(drop=True)