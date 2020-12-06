from rouge import Rouge  # pip install --user rouge
import textrank
import cluster
import pandas as pd

def get_rouge(pred, ref):
    rouge = Rouge()
    scores = rouge.get_scores(pred, ref)
    print(scores[0])
    return scores[0]

def rouge_overall_textrank(text_df, sum_df, embeddings):
    rouge = Rouge()
    texts = [t for t in text_df["text"]]
    sums = [s for s in sum_df["text"]]
    pred_sums = textrank.get_summary(texts, embeddings, 3)
    scores = rouge.get_scores(pred_sums, sums, avg = True)
    print(scores)
    return scores

def rouge_overall_cluster(text_df, sum_df):
    rouge = Rouge()
    texts = [t for t in text_df["text"]]
    sums = [s for s in sum_df["text"]]
    pred_sums = cluster.get_summary(texts, 3)
    scores = rouge.get_scores(pred_sums, sums, avg = True)
    print(scores)
    return scores


def get_barchart(textrank_score, cluster_score, scoretype):
    tr = [x for x in textrank_score[scoretype].values()]
    cl = [x for x in cluster_score[scoretype].values()]
    index = ["f", "p", "r"]
    df = pd.DataFrame({'textrank':tr, "cluster": cl}, index = index)
    ax = df.plot.bar(rot=1)
    pass