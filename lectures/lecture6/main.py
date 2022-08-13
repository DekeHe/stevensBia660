from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
def f(str):
    ap,al=[],[]
    with open(str) as ti:
        for str in ti:
            l=str.strip().split('\t')
            ap.append(l[0].lower())
            al.append(int(l[1]))
    print(ap)
    return [ap,al]
##

[ap,al]=f('a.txt')
[bp,bl]=f('b.txt')

cv=CountVectorizer()
cv.fit(ap)
am=cv.transform(ap)
bm=cv.transform(bp)

cl=DecisionTreeClassifier()
cl.fit(am,al)

predictArray=cl.predict(bm)

print (accuracy_score(predictArray,bl))


