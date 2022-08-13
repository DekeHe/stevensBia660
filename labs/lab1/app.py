#print the frequencies of a page

import requests
import re
import nltk
from nltk.corpus import stopwords
stopwords=stopwords.words('english')
from operator import itemgetter

def get():
    result=[]
    url='http://www.uazone.com/gis/022098fedreg.txt'
    text=requests.get(url).text
    text=re.sub('[^a-z]',' ',text.lower())
    list=text.split(' ')
    d={}
    for v in list:
        if v not in stopwords and v!='':
            if v in d:d[v]+=1
            else: d[v]=1
    result=sorted(d.items(),key=itemgetter(1),reverse=True)
    return result

print(get())