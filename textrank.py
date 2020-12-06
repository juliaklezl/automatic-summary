# imports
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# this program is based on the algorithm outlined in Prateek Joshi's tutorial https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/. 

def get_sentence_list(textstring):  # split text into sentences, get a list without stopwords for the similarity calculation and a complete list for assembling the summary in the end
    sentences = sent_tokenize(textstring)
    lower_sentences = [s.lower() for s in sentences]
    stop_words = stopwords.words('english')
    nostop_sentences = []
    for s in lower_sentences:
        new_s = " ".join([w for w in s.split() if w not in stop_words])
        nostop_sentences.append(new_s)
    return sentences, nostop_sentences

def get_embeddings(embed_file): # load pre-trained GloVe embeddings
    word_embeddings = {}
    with open(embed_file, encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = vec
    print("loaded embeddings")
    return word_embeddings


def sent_embeddings(sent_list, embed_dict): # get sentence embeddings by taking average embedding of all words of a sentence
    sentence_vectors = []
    for s in sent_list:
        if len(s) > 1: 
            sent_vectors = [embed_dict.get(w, np.zeros((100,))) for w in s.split()]
            num_words = len(s.split())
            v = sum(sent_vectors)/(num_words)
        else:  
            v = np.zeros((100,)) 
        sentence_vectors.append(v)
    return sentence_vectors
    
def similarity_matrix(textstring, embeddings): # calculate cosine similarity between all sentences in the text
    sents, c_sents =  get_sentence_list(textstring)
    embeds = sent_embeddings(c_sents, embeddings)
    sim_mat = np.zeros([len(sents), len(sents)])
    for i in range(len(sents)):
        for j in range(len(sents)):
            if i != j:  
                e_i = embeds[i].reshape(1,100)
                e_j = embeds[j].reshape(1,100)
                sim_mat[i][j] = cosine_similarity(e_i, e_j)
    return sim_mat, sents

def get_summary(textstrings, embeddings, len_sum): # create graph based on similarities, apply pagerank algorithm, and pick n most important sentences
    summaries = []
    for textstring in textstrings:
        sm, sents = similarity_matrix(textstring, embeddings)
        graph = nx.from_numpy_array(sm)
        try:
            scores = nx.pagerank(graph)
            scores_sents = ((scores[i],s) for i,s in enumerate(sents))
            ranked_sentences = sorted(scores_sents, reverse=True)
            summary = ""
            for i in range(len_sum):
                if len(ranked_sentences) > i:  
                    summary += ranked_sentences[i][1] + " "
            print(summary)
        except:   # if pagerank doesn't converge, first n sentences are taken as summary
            summary = " ".join(sents[:len_sum])
        summaries.append(summary)
    return summaries

#def get_summary(textstrings, embeddings, len_sum): # create graph based on similarities, apply pagerank algorithm, and pick n most important sentences
#    summaries = []
#    for textstring in textstrings:
#        sm, sents = similarity_matrix(textstring, embeddings)
#        graph = nx.from_numpy_array(sm)
#        scores = nx.pagerank(graph)
#        scores_sents = ((scores[i],s) for i,s in enumerate(sents))
#        ranked_sentences = sorted(scores_sents, reverse=True)
#        summary = ""
#        for i in range(len_sum):
#            if len(ranked_sentences) > i:  
#                summary += ranked_sentences[i][1] + " "
#        print(summary)
#        summaries.append(summary)
#    return summaries
    

    