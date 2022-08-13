import csv
import nltk
from nltk import load
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import sent_tokenize,word_tokenize


# function that loads a lexicon of positive words to a set and returns the set
def loadLexicon(fname):
    newLex = set()
    lex_conn = open(fname)

    # add every word in the file to the set
    for line in lex_conn:
        newLex.add(line.strip())  # remember to strip to remove the lin-change character
    lex_conn.close()

    return newLex


def getOpinions(input_file, feature_num):
    # load the positive and negative lexicons into sets
    posLex = loadLexicon('positive-words.txt')
    negLex = loadLexicon('negative-words.txt')

    polarity_count = {}  # maps each noun to the number of times it appears in a negative sentence and to the number of times it appears in a positive sentence

    fin = open(input_file, encoding='utf8')

    reader = csv.reader(fin)

    for line in reader:  # for each review

        text, rating = line  # get the text and rating

        sentences = sent_tokenize(text)  # split the review into sentences

        for sentence in sentences:  # for each sentence

            words = word_tokenize(sentence)  # split the review into words

            tagged_words = nltk.pos_tag(words)  # POS tagging for the words in the sentence

            nouns_in_sentence = set()  # set of all the nouns in the sentence

            positive_word_count = 0  # number of positive words in the sentence
            negative_word_count = 0  # number of negative words in the sentence

            # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
            for tagged_word in tagged_words:

                if tagged_word[1].startswith('NN'):  # if the word is a noun

                    noun = tagged_word[0].lower()  # lower case the noun

                    if len(noun) < 3: continue  # ignore nouns with less than 3 characters

                    nouns_in_sentence.add(noun)  # add the noun to the set

                    if noun not in polarity_count:  # first time we see this noun
                        polarity_count[noun] = [0, 0]  # positives, negatives

                if tagged_word[1].startswith('JJ') and tagged_word[
                    0] in posLex:  # if the word is in the positive lexicon
                    positive_word_count += 1

                elif tagged_word[1].startswith('JJ') and tagged_word[
                    0] in negLex:  # if the word is in the negative lexicon
                    negative_word_count += 1

            sentence_polarity = positive_word_count - negative_word_count

            if sentence_polarity > 0:  # positive sentence
                for noun in nouns_in_sentence:  # for each noun that we found in the sentence
                    polarity_count[noun][0] += 1  # increase the positive count

            elif sentence_polarity < 0:  # negative sentence
                for noun in nouns_in_sentence:  # for each noun that we found in the sentence
                    polarity_count[noun][1] += 1  # increase the negative count

    fin.close()

    # sort noun based on their total polarity counts (pos+neg)
    sorted_polarity_count = sorted(polarity_count.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)

    """

    amazing->[30,5]
    horrible->[4,35]

    [ ('amazing',[30,5]),  ('horrible',[4,35]) ]

    30+5, 4+35  
    35,40

    """

    # get the top feature_num features
    top = sorted_polarity_count[:feature_num]

    return top

result=getOpinions('amazonreviews.csv',15)

for noun in result:
    print(noun)

D={'battery':[10,5],'quality':[4,2],'price':[3,8]}
sortedD=sorted(D.items(),key=lambda x:x[1][0]+x[1][1],reverse=True)
print(sortedD)

_POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
tagger = load(_POS_TAGGER)

sent='My name is Ted'
words=word_tokenize(sent)
tagged_words=tagger.tag(words)
print(tagged_words)

import nltk
nltk.download()