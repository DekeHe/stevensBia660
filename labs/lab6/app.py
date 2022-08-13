from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# sudo pip install numpy==1.17.4

def getL(str):
    reviews,labels=[],[]
    f=open(str,'r')
    for v in f:
        t=v.split('\t')
        reviews.append(t[0].lower())
        labels.append(int(t[1]))
    return reviews,labels

reviews_train, labels_train=getL('reviews_train.txt')
reviews_test,labels_test=getL('reviews_test.txt')

counter=CountVectorizer()
counter.fit(reviews_train)

counts_train=counter.transform(reviews_train)
counts_test=counter.transform(reviews_test)

clf=MLPClassifier(activation = 'tanh', solver='adam', alpha=1e-4,
                    hidden_layer_sizes=(128,128,), random_state=1 )

clf.fit(counts_train,labels_train)


pred=clf.predict(counts_test)

print(accuracy_score(pred,labels_test))

