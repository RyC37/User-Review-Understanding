#!/usr/bin/env python
# Author: Sicong Zhao

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import re
import numpy as np
!pip install hdbscan
import hdbscan

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

def cluster(model, df, threshold, thrd, positive=True, components=0, method='HBDSCAN'):
    '''
    Clustering embedded sentences.
    
    Parameters:
        model       (obj):   The sentence embedding model.
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
    embed = model.encode(df_['review'].values)
    # Normalize embedding
    f = np.array(embed)
    embed = (f - f.mean(axis=0)) / f.std(axis=0)
    # PCA
    if components > 0:
        x = pd.DataFrame(embed)
        pca = PCA(n_components=components)
        embed = pca.fit_transform(x)
    # Clustering
    if method == 'AC':
        clust = AgglomerativeClustering(n_clusters=None,distance_threshold=threshold).fit(embed)
    elif method == 'HDBSCAN':
        clust = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, alpha=1.3).fit(embed)
    df_['cluster'] = clust.labels_
    return df_.sort_values(by='cluster').reset_index(drop=True)