# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""




"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    #print(len(X),'X')
    pos_vocab = Counter()
    neg_vocab = Counter()
   
    for i in range(len(X)):
        if y[i] > 0:
            pos_vocab.update(X[i])
        else:
            neg_vocab.update(X[i])
    return pos_vocab, neg_vocab


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word pair appears 
    """
    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}
    ##TODO:
    raise RuntimeError("Replace this line with your code!")
    return dict(pos_vocab), dict(neg_vocab)


# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)
    pos_vocab, neg_vocab = create_word_maps_uni(train_set, train_labels)

    pos_sum = sum(pos_vocab.values())
    neg_sum = sum(neg_vocab.values())
    pos_prior = pos_sum / (pos_sum + neg_sum)

    # for key, value in pos_vocab.items():
    #     p_w_given_pos[key] = (laplace + value) / (pos_sum + laplace*(1 + len(pos_vocab)))
    # for key, value in neg_vocab.items():
    #     p_w_given_neg[key] = (laplace + value) / (neg_sum + laplace*(1 + len(neg_vocab)))
    laplace = 1
    no_word_pos = laplace / (pos_sum + laplace*(1 + len(pos_vocab)))
    no_word_neg = laplace / (neg_sum + laplace*(1 + len(neg_vocab)))
    def estimate(words):
        p_pos = np.log(pos_prior)
        p_neg = np.log(1 - pos_prior)
        for word in words:
            if word in pos_vocab:
                lp = (laplace + pos_vocab[word]) / (pos_sum + laplace*(1 + len(pos_vocab)))
                p_pos += np.log(lp)
            else:
                p_pos += np.log(no_word_pos)

            if word in neg_vocab:
                lp = (laplace + neg_vocab[word]) / (neg_sum + laplace*(1 + len(neg_vocab)))
                p_neg += np.log(lp)
            else: 
                p_neg += np.log(no_word_neg)
        
        return (p_pos > p_neg)

    dev_set_labels = [] 
    for email in dev_set:
        for w in email:
            if w not in pos_vocab or w not in neg_vocab:
                laplace += 1

    print("laplace", laplace)     
    for email in dev_set:
        if estimate(email):
            dev_set_labels.insert(0, 1)
        else:
            dev_set_labels.insert(0, 0) 

    return dev_set_labels

# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    max_vocab_size = None

    raise RuntimeError("Replace this line with your code!")

    return []
