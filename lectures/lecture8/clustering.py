#https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe/data#
from nltk.tokenize import word_tokenize
import numpy
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import fisher_exact as FE
import csv


#function that loads a lexicon of positive words to a set and returns the set
def loadLexicon(fname):
    newLex=set()
    lex_conn=open(fname)    #add every word in the file to the set
    for line in lex_conn:
        newLex.add(line.strip())# remember to strip to remove the lin-change character
    lex_conn.close()
    return newLex


# compute distance between 2 reviews based on their sentiment
def senti_dist(r1, r2, posLex, negLex):
    pos_count1, pos_count2, neg_count1, neg_count2 = 0, 0, 0, 0  # positive and negative counts for the 2 reviews

    for term in r1:  # for each term in r1
        if term in posLex:
            pos_count1 += 1
        elif term in negLex:
            neg_count1 += 1

    for term in r2:  # for each term in r2
        if term in posLex:
            pos_count2 += 1
        elif term in negLex:
            neg_count2 += 1

    # compute the sentiment score for r1 and r2
    sent_score1 = (pos_count1 - neg_count1) / (pos_count1 + neg_count1 + 1)
    sent_score2 = (pos_count2 - neg_count2) / (pos_count2 + neg_count2 + 1)

    sent_dist = abs(sent_score1 - sent_score2) / 2  # combine the 2 scores to compute their senti distance

    return sent_dist


def load_reviews(review_file):
    f = open(review_file)
    f.readline()  # skip header line

    reviews = []  # review texts
    scores = []  # score texts

    reader = csv.reader(f)
    for review, score in reader:  # for each review
        reviews.append(review)
        scores.append(float(score))

    f.close()
    return reviews, scores


def create_dist_matrix(reviews):
    # load the positive and negative lexicons into sets
    posLex = loadLexicon('positive-words.txt')
    negLex = loadLexicon('negative-words.txt')

    N = len(reviews)

    # square distance matrix full of zeros
    sdist = numpy.zeros(shape=(N, N))

    terms_per_review = []

    for i in range(N):  # for each review
        if i % 50 == 0: print(i, 'reviews loaded')
        terms1 = word_tokenize(reviews[i].lower())  # tokenize the first review
        terms_per_review.append(terms1)
        for j in range(i + 1, N):  # for each other  review

            terms2 = word_tokenize(reviews[j].lower())  # tokenize the second review

            sdist[i][j] = senti_dist(terms1, terms2, posLex, negLex)  # compute the distance
            sdist[j][i] = sdist[i][j]  # distance is symmetric

    return sdist, terms_per_review


"""
Perform the clustering step
"""


def get_clusters(reviews_file):
    reviews, scores = load_reviews(reviews_file)

    sdist, terms_per_review = create_dist_matrix(reviews)

    # cluster the reviews based on distance
    clustering = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average').fit(sdist)

    return clustering, scores, terms_per_review


clustering, scores, terms_per_review = get_clusters('reviews.csv')

# compute the average review score for cluster0
cluster0_scores = []
for i in range(len(scores)):
    if clustering.labels_[i] == 0:
        cluster0_scores.append(scores[i])
print('CLUSTER 0:', numpy.mean(cluster0_scores))

# compute the average review score for cluster1
cluster1_scores = []
for i in range(len(scores)):
    if clustering.labels_[i] == 1:
        cluster1_scores.append(scores[i])
print('CLUSTER 1:', numpy.mean(cluster1_scores))

print(clustering.labels_)



"""
Look for characteristic terms for each of the 2 clusters using the fisher test:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
"""

def find_key_terms(clustering,terms_per_review):

    cluster0_term_freq={}
    cluster1_term_freq={}
    allterms=set()
    N0,N1=0,0 # total cumulative frequency of all terms in each cluster

    for i in range(len(clustering.labels_)): # for each review
        if clustering.labels_[i]==0:
            for term in terms_per_review[i]:
                cluster0_term_freq[term]=cluster0_term_freq.get(term,0)+1
                allterms.add(term)
                N0+=1
        else:
            for term in terms_per_review[i]:
                cluster1_term_freq[term]=cluster1_term_freq.get(term,0)+1
                allterms.add(term)
                N1+=1


    cluster0_distintive_terms=[]
    cluster1_distintive_terms=[]

    for term in allterms:
        freq0=cluster0_term_freq.get(term,0)
        freq1=cluster1_term_freq.get(term,0)

        score,pval=FE([[freq0,freq1],[N0-freq0,N1-freq1]])

        if pval<=0.01:
            ratio0=freq0/N0
            ratio1=freq1/N1

            if ratio0>ratio1:
                cluster0_distintive_terms.append(term)
            else:
                cluster1_distintive_terms.append(term)

    print('CLUSTER 0 DISTINCTIVE TERMS:')
    print(cluster0_distintive_terms)

    print()

    print('CLUSTER 1 DISTINCTIVE TERMS:')
    print(cluster1_distintive_terms)

find_key_terms(clustering,terms_per_review)