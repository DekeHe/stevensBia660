"""
A function that reads a file from the web and returns the most frequent word in the file
"""

import re
import requests
from operator import itemgetter


def run(url):
    freq = {}  # keep the freq of each word in the file

    stopLex = set()  # create an empty set of stopwords
    fin = open('stopwords.txt')  # open the stopwords file
    for line in fin:  # for each line
        stopLex.add(line.strip())  # strip the line to remove white space at the beginning and end of the line
    fin.close()  # close the connection to the file

    for i in range(5):  # try 5 times

        # send a request to access the url
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
        if response:  # explanation on response codes: https://realpython.com/python-requests/#status-codes
            break  # we got the file, break the loop
        else:
            print('failed attempt', i)

    # all five attempts failed, return  None
    if not response: return None

    text = response.text  # read in the text from the file

    text = re.sub('[^a-z]', ' ', text.lower())  # replace all non-letter characters  with a space

    words = text.split(' ')  # split to get the words in the text

    for word in words:  # for each word in the sentence
        if word == '' or word in stopLex:
            continue  # ignore empty words and stopwords
        else:
            if word in freq:  # we have seen this word before
                freq[word] = freq[word] + 1  # add 1
            else:  # first time we see the word
                freq[word] = 1  # initialize the freq to 1

    freq_values = freq.values()

    sorted_words = sorted(freq, key=itemgetter(1), reverse=)

    return freq

print(run('http://www.uazone.com/gis/022098fedreg.txt'))