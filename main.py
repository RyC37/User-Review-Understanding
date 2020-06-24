#!/usr/bin/env python
# Author: Sicong Zhao

import argparse
from utils import *
from models import InferSent
from textblob import TextBlob
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--fpath', type=str, default='', help='The relative path of data.')
parser.add_argument('--cuda', type=bool, default=False, help='Decide use cuda or not.')
parser.add_argument('--model_version', type=int, default=1, help='Model version, 1 or 2.')
opt = parser.parse_args()

# Load model
model_version = opt.model_version
MODEL_PATH = "encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.cuda() if opt.cuda else model
W2V_PATH = 'GloVe/glove.840B.300d.txt' if opt.model_version == 1 else 'fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)
model.build_vocab_k_words(K=100000)

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
pos = cluster(model, rp_df,3,0.2,True)
neg = cluster(model, rp_df,3,-0.2,False)
pos.to_csv('output/pos_review_clustered.csv',index=False)
neg.to_csv('output/neg_review_clustered.csv',index=False)

