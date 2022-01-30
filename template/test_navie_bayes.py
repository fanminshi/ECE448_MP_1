import naive_bayes as nb

# python3 mp1.py --training ../data/spam_data/train --development ../data/spam_data/dev --stemming False --lowercase False --laplace 1.0 --pos_prior 0.8
train_set, train_labels, dev_set, dev_labels = nb.load_data("data/spam_data/train","data/spam_data/dev",False, False)
# p , n = nb.create_word_maps_uni(train_set, train_labels)
# print(p, n)
estimates = nb.naiveBayes(train_set, train_labels, dev_set)
s =  sum([abs(ele) for ele in estimates - dev_labels])
print(1 - (s / len(dev_set)))