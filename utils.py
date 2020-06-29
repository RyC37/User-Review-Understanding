#!/usr/bin/env python
# Author: Sicong Zhao

import pandas as pd
from sklearn.cluster import AgglomerativeClustering

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


def cluster(model, df, threshold, thrd, positive=True):
    '''
    Clustering embedded sentences.
    
    Parameters:
        model       (obj): The sentence embedding model.
        df          (obj): The dataframe that contains reviews.
        threshold   (float): The threshold for clustering distance.
        thrd        (float): The threshold for deciding if a review is positive or negative.
        positive    (bool): Decide if clustering positive reviews or negative reviews.
    Return: 
        The dataframe with cluster id assigned to each review.
    '''
    if positive:
        df_ = df[df['polarity'] > thrd].copy()
    else:
        df_ = df[df['polarity'] < thrd].copy()
    embed = model.encode(df_['review'].values)
    clust = AgglomerativeClustering(n_clusters=None,distance_threshold=threshold).fit(embed)
    df_['cluster'] = clust.labels_
    return df_.sort_values(by='cluster').reset_index(drop=True)