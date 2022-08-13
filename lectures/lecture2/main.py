# Read this first: https://www.w3schools.com/html/default.asp
from bs4 import BeautifulSoup
import re
import time
import requests
import csv


def run(url):
    fw = open('reviews.txt', 'w', encoding='utf8')  # output file

    writer = csv.writer(fw, lineterminator='\n')  # create a csv writer for this file

    for i in range(5):  # try 5 times
        # send a request to access the url
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36', })
        if response:  # explanation on response codes: https://realpython.com/python-requests/#status-codes
            break  # we got the file, break the loop
        else:
            print('fail', i)
            time.sleep(2)  # wait 2 secs

    # all five attempts failed, return  None
    if not response: return None

    html = response.text  # read in the text from the file

    soup = BeautifulSoup(html, features='html.parser')  # parse the html

    reviews = soup.findAll('div', {'class': re.compile('review_table_row')})  # get all the review divs

    for review in reviews:

        critic, text = 'NA', 'NA'  # initialize critic and text
        criticChunk = review.find('a', {'href': re.compile('/critics/')})
        if criticChunk: critic = criticChunk.text.strip()

        textChunk = review.find('div', {'class': 'the_review'})
        if textChunk: text = textChunk.text.strip()

        writer.writerow([critic, text])  # write to file

    fw.close()

url='https://www.rottentomatoes.com/m/space_jam/reviews/'
run(url)

