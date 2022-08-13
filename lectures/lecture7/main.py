from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")



#read the reviews and their polarities from a given file
def loadData(fname):
    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:
        review,rating=line.strip().split('\t')
        reviews.append(review.lower())
        labels.append(rating)
    f.close()
    return reviews,labels


rev_train,labels_train=loadData('reviews_train.txt')
rev_test,labels_test=loadData('reviews_test.txt')

#Build a counter based on the training dataset
counter = CountVectorizer(stop_words=stopwords.words('english'))
counter.fit(rev_train)

#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(rev_train)#transform the training data
counts_test = counter.transform(rev_test)#transform the testing data


model1 = LogisticRegression(random_state=1)
model2 = RandomForestClassifier(random_state=1)
model3 = MLPClassifier(hidden_layer_sizes=(128, 128,), random_state=1)
model4 = SVC(probability=True, random_state=1)
model5 = AdaBoostClassifier(random_state=1)


model1_grid = [{'penalty': ['none','l1', 'l2', 'elasticnet'],
                'C': [0.5, 1, 1.5, 2]}]
gridsearch_model1 = GridSearchCV(model1, model1_grid, cv=5)
gridsearch_model1.fit(counts_train,labels_train)
print(gridsearch_model1.best_params_)

model2_grid = [{'n_estimators': [100, 200, 300],
                'max_depth': [10,11,12,13,14,15],
                'criterion': ['gini', 'entropy', 'log_loss']}]
gridsearch_model2 = GridSearchCV(model2, model2_grid, cv=5)
gridsearch_model2.fit(counts_train,labels_train)
print(gridsearch_model2.best_params_)

model3_grid = [{'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'adam']}]
gridsearch_model3 = GridSearchCV(model3, model3_grid, cv=5)
gridsearch_model3.fit(counts_train,labels_train)
print(gridsearch_model3.best_params_)

model4_grid = [{'C': [0.5, 1, 1.5, 2],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]
gridsearch_model4 = GridSearchCV(model4, model4_grid, cv=5)
gridsearch_model4.fit(counts_train,labels_train)
print(gridsearch_model4.best_params_)

model5_grid = [{'n_estimators' : [50, 100, 150, 200],
                'algorithm' : ['SAMME', 'SAMME.R']}]
gridsearch_model5 = GridSearchCV(model5, model5_grid, cv=5)
gridsearch_model5.fit(counts_train,labels_train)
print(gridsearch_model5.best_params_)



model1N = LogisticRegression(C=0.5, penalty='l2', random_state=1)
model2N = RandomForestClassifier(criterion='gini', max_depth=14, n_estimators=200, random_state=1)
model3N = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(128, 128,), random_state=1)
model4N = SVC(C=2, kernel='rbf', probability=True, random_state=1)
model5N = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=200, random_state=1)

VT = VotingClassifier(estimators=[('lr', model1N),
                                  ('rf', model2N),
                                  ('ann', model3N),
                                  ('svc', model4N),
                                  ('abc', model5N)],
                      voting='soft')

VT.fit(counts_train,labels_train)

#use the VT classifier to predict
predicted=VT.predict(counts_test)

#print the accuracy
print(accuracy_score(predicted,labels_test))

