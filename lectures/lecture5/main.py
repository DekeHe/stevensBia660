#https://bab2min.github.io/tomotopy/v0.12.2/en/
import tomotopy as tp
# pip install --upgrade pip
# pip3 install tomotopy
from nltk.corpus import stopwords
import re


def tokenizer(doc, sw):
    return [word for word in [re.sub('[^a-z]', '', x.lower()) for x in doc.strip().split()] if
            word not in sw and len(word) > 2]

#stopwords
sw=stopwords.words('english')

# new LDA model
mdl = tp.LDAModel(k=20)

# add the docs to the model
with open('news.txt') as f:
    for doc in f:
        mdl.add_doc(tokenizer(doc, sw))

# train LDA model
for i in range(0, 500, 10):
    mdl.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

# print doc info
for doc in mdl.docs:
    print([round(p, 2) for p in doc.get_topic_dist()])
    print()

# print topic info
for k in range(mdl.k):
    topk_words = [pair[0] for pair in mdl.get_topic_words(k, top_n=15)]
    print(k, topk_words)
    print()

# new LDA model
mdl = tp.CTModel(k=20)

# add the docs to the model
with open('news.txt') as f:
    for doc in f:
        mdl.add_doc(tokenizer(doc, sw))

# train LDA model
for i in range(0, 500, 10):
    mdl.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

# print topic info
for k in range(mdl.k):

    topk_words = [pair[0] for pair in mdl.get_topic_words(k, top_n=15)]
    print(k, topk_words)

    # get correlations
    corrs = mdl.get_correlations(k)
    for i in range(20):
        if i != k and corrs[i] > 0.3:
            print('Related Topic:{} ({:.2f})'.format(i, corrs[i]))

    print()

# new LDA model
mdl = tp.MGLDAModel(k_g=20, k_l=40)

c = 0
# add the docs to the model
with open('hotelreviews.csv') as f:
    for doc in f:
        mdl.add_doc(tokenizer(doc.split('\t')[0], sw))
        if c == 50000: break

# train LDA model
for i in range(0, 500, 10):
    mdl.train(10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

# print topic info
for k in range(mdl.k_g):  # mdl.k_,mdl.k_g+mdl.k_l):

    topk_words = [pair[0] for pair in mdl.get_topic_words(k, top_n=15)]
    print(k, topk_words)
    print()








##new
from nltk.corpus import stopwords

import tomotopy
import numpy
import builtins
import  re
##
stopwords=stopwords.words('english')
# print(stopwords)

def f(str):
    r=[]
    a=str.strip().split()
    for i in range(len(a)):
        a[i]=re.sub('[^a-z]','',a[i].lower())
    # print(a)
    for v in a:
        if v not in stopwords and len(v)>2:
            r.append(v)
    return r

lm=tomotopy.MGLDAModel(k_g=10,k_l=5)

i=0
with builtins.open('news.txt') as ti:


    r=[]
    for str in ti:
        i+=1
        if i==10:break
        lm.add_doc(f(str))


for i in range(0,500,10):
    lm.train(10)
    # print(i,lm.ll_per_word)

# print(len(lm.docs))
# for v in lm.docs:
#     for p in v.get_topic_dist():
#         print(round(p,2))

# print(lm)

for i in range(lm.k):
    r=[]
    temp=lm.get_topic_words(i,top_n=5)
    for v in temp:
        r.append( v[0])
    print(r)
    # print()




