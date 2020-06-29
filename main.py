#!/usr/bin/env python
# Author: Sicong Zhao

import argparse
from utils import *
from textblob import TextBlob
import itertools
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--fpath', type=str, default='../../data/snapchat_reviews.csv', help='The relative path of data.')
parser.add_argument('--cuda', type=bool, default=False, help='Decide use cuda or not.')
opt = parser.parse_args()

# Load model
model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')

# Load data
reviews = load_data(opt.fpath)
rs = reviews['Review Text'].sample(n=2000).str.split(pat='.').values
rs = [r for r in rs if type(r) != float]
rl = list(itertools.chain.from_iterable(rs))
rp = [st for st in [preprocess(s) for s in rl] if st != None]
rp = list(set(rp)) # Remove duplications

# Add sentiment
rp_sentiment = []
for r in rp:
    testimonial = TextBlob(r)
    rp_sentiment.append([r,testimonial.sentiment.polarity,testimonial.sentiment.subjectivity])
rp_df = pd.DataFrame(rp_sentiment, columns=['review','polarity','subjectivity'])

# Embedding
pos = cluster(model, rp_df, 20, 0.2, True)
neg = cluster(model, rp_df, 20, -0.2, False)
pos.to_csv('result/pos_review_clustered.csv',index=False)
neg.to_csv('result/neg_review_clustered.csv',index=False)