import skipthoughts  # pip install skipthoughts
from nltk.tokenize import sent_tokenize
from torch.autograd import Variable
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# this program is based on Kushal Chauhan's tutorial: https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1. A few lines of the get_summary function are verbatim copies.

def get_sentence_list(textstring):
    sentences = sent_tokenize(textstring)
    lower_sentences = [s.lower() for s in sentences]
    max_len = max([len(s.split(" ")) for s in lower_sentences])
    return lower_sentences, max_len

def get_vocab(sent_list):
    vocab = []
    for s in sent_list:
        words = s.split(" ")
        for word in words:
            if word not in vocab:
                vocab.append(word)
    vocab.append("paddingxxx")
    return vocab

def get_sent_embs(textstring):
    dir_str = 'data/skip-thoughts'
    sent_l, max_len = get_sentence_list(textstring)
    vocab = get_vocab(sent_l)
    new_sent_l = []
    for sent in sent_l:  
        words = sent.split(" ")
        s = [vocab.index(word) for word in words]  # convert text to number encodings
        while len(s) < max_len:  # padding to max sentence length of text
            s.append(vocab.index("paddingxxx"))
        new_sent_l.append(s)
    model = skipthoughts.UniSkip(dir_str, vocab) # initialize skipthoughts
    inp = Variable(torch.LongTensor(new_sent_l)) 
    sent_embs = model(inp)  
    return sent_embs

def get_summary(textstring, len_sum): 
    sent_embs = get_sent_embs(textstring).detach().numpy()
    sent_l = get_sentence_list(textstring)
    kmeans = KMeans(n_clusters=len_sum)
    kmeans = kmeans.fit(sent_embs)  
    avg = []
    for j in range(len_sum):
        i = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(i))  # get average position of sentences in text for each cluster
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, sent_embs) # # get most central sentence for every cluster
    ordering = sorted(range(len_sum), key=lambda k: avg[k])   # # sort clusters by their average position in text
    summary = ' '.join([sent_l[0][closest[i]] for i in ordering])   # # combine most central sentence of each cluster to summary
    print(summary)
    return summary  