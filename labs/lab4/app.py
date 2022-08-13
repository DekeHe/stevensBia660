import csv
from nltk import pos_tag
from nltk import sent_tokenize,word_tokenize
from operator import itemgetter

def getGl(str):
    s=set()
    f=open(str,'r')
    for v in f:s.add(v.strip())
    f.close()
    return s
gl=getGl('positive-words.txt')
bl=getGl('negative-words.txt')

def get():
    r={}
    f=open('amazonreviews.csv','r')
    reader=csv.reader(f,lineterminator='\n')

    for v in reader:
        ss=sent_tokenize(v[0])
        for s in ss:
            color=0
            t=word_tokenize(s)
            words=pos_tag(t)
            nouns=set()
            for v in words:
                if v[1].startswith('NN'):
                    curNoun=v[0].lower()
                    nouns.add(curNoun)
                    if curNoun not in r:r[curNoun]=[0,0]
            for v in words:
                if v[1].startswith('JJ'):
                    if v[0] in gl:color+=1
                    if v[0] in bl:color-=1
            if color>0:
                for v in nouns:r[v][0]+=1
            if color<0:
                for v in nouns:r[v][1]+=1
    r=sorted(r.items(),key=itemgetter(1),reverse=True)
    print(r)
get()



