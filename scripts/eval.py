import torch
import torch.nn as nn
import numpy as np
import sys
import os

def get_eval(data_dict):
    """ Evaluation of the model
    Parameters
    ----------
    data_dict: dict includes objects/relationship predict and triplets

    Returns: dict includes eval metrics
    -------
    """
    batch_size = data_dict["objects_id"].size(0)
    top5_ratio_o = []
    top10_ratio_o = []
    top3_ratio_r = []
    top5_ratio_r = []
    top50_predicate = []
    top100_predicate = []
    for i in range(batch_size):
        object_pred = data_dict["objects_predict"][i]
        object_cat = data_dict["objects_cat"][i]

        predicate_pred = data_dict["predicate_predict"][i]
        triples = data_dict["triples"][i]
        predicate_cat = triples[:,2]

        top5_ratio_o.append(topk_ratio(object_pred, object_cat, 5))
        top10_ratio_o.append(topk_ratio(object_pred, object_cat, 10))
        top3_ratio_r.append(topk_ratio(predicate_pred, predicate_cat, 3))
        top5_ratio_r.append(topk_ratio(predicate_pred, predicate_cat, 5))

        edges = data_dict["edges"][i]  # store the index
        object_conf, _ = torch.topk(object_pred, 1, 1)
        object_conf = object_conf.squeeze(-1)
        for x in range(predicate_pred.size(0)):
            for y in range(predicate_pred.size(1)):
                if x==0 and y==0:
                    triple_scores = np.array([[object_conf[edges[x,0]] * object_conf[edges[x,1]] * predicate_pred[x,y], triples[x,0], triples[x,1], y]])
                else:
                    t = np.array([[object_conf[edges[x,0]] * object_conf[edges[x,1]] * predicate_pred[x,y], triples[x,0], triples[x,1], y]])
                    triple_scores = np.concatenate((triple_scores,t), axis=0)
        triple_scores = triple_scores[(-triple_scores[:,0]).argsort()]  # descending order

        top50_predicate.append(topk_triplet(triple_scores, triples, 50))
        top100_predicate.append(topk_triplet(triple_scores, triples, 100))

    data_dict["top5_ratio_o"] = np.mean(np.array(top5_ratio_o))
    data_dict["top10_ratio_o"] = np.mean(np.array(top10_ratio_o))
    data_dict["top3_ratio_r"] = np.mean(np.array(top3_ratio_r))
    data_dict["top5_ratio_r"] = np.mean(np.array(top5_ratio_r))
    data_dict["top50_predicate"] = np.mean(np.array(top50_predicate))
    data_dict["top100_predicate"] = np.mean(np.array(top100_predicate))

    return data_dict


def topk_ratio(logits, category, k):
    """
    Parameters
    ----------
    logits: [N C] N objects/relationships with C categroy
    category: [N 1] N objects/relationships
    k:  top k

    Returns: recall of top k (R@k)
    -------
    """
    _, topk_pred = torch.topk(logits, k, dim=1)
    topk_ratio = 0
    for index, x in enumerate(topk_pred):
        if category[index] in x:
            topk_ratio += 1
    topk_ratio /= topk_pred.size(0)
    return topk_ratio

def topk_triplet(score, triples, k):
    """
    Parameters
    ----------
    score: multiplying results of each probability of triplets
    triples: triplets exist in the scene
    k:  top k

    Returns: recall of top k (R@k)
    -------
    """
    tri = score[:,1:]
    triplets = triples.cpu().tolist()
    assert len(tri)>=k
    ratio = 0
    for i in range(k):
        s = tri[i,0].cpu().item()   # make every item on the same device
        o = tri[i,1].cpu().item()
        p = tri[i,2]
        t = [s, o, p]
        if t in triplets:
            ratio += 1
    ratio /= k
    return ratio