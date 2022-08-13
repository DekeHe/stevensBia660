from nltk.corpus import stopwords
import tomotopy as tp
import re
sw=stopwords.words('english')

def tokenize(p):
    r=[]
    for v in p.strip.split():
        v=re.sub('[^a-z]',' ',v.lower())
        if v not in sw:r.append(v)
mdl=tp.LD
with open('news.txt') as f:
    for p in f:
        mdl.add_doc()
