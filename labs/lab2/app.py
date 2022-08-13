#write the page chunks

import csv
import requests
from bs4 import BeautifulSoup
import re

def write():
    f=open('result.csv','w')
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(['name','freshOrRotten'])

    url='https://www.rottentomatoes.com/m/official_competition/reviews'
    text=requests.get(url).text
    bs=BeautifulSoup(text,features='html.parser')
    a=bs.findAll('div',{'class':re.compile('row review_table_row')})
    for v in a:
        r=[]
        s=v.findNext('a',{'href':re.compile('/critics/')}).text
        r.append(s)

        temp=v.findNext('div', {'class':re.compile('review_icon icon small ')})
        s=temp['class'][-1]
        r.append(s)

        writer.writerow(r)

write()

