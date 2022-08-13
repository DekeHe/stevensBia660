import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
def getL(str):
    reviews,labels=[],[]
    f=open(str,'r')
    for v in f:
        t=v.split('\t')
        reviews.append(t[0])
        labels.append(t[1])
    return reviews,labels
reviews_train,labels_train=getL('reviews_train.txt')
reviews_test,labels_test=getL('reviews_test.txt')

counter=CountVectorizer()

counter.fit(reviews_train)

counts_train=counter.transform(reviews_train)
counts_test=counter.transform(reviews_test)

model1=LogisticRegression(random_state=1)
model2=RandomForestClassifier(random_state=1)
model3=MLPClassifier(hidden_layer_sizes=(128, 128,), random_state=1)
model4=SVC(probability=True, random_state=1)
model5=AdaBoostClassifier(random_state=1)

model1_grid=[{'penalty': ['none','l1', 'l2', 'elasticnet'],
'C': [0.5, 1, 1.5, 2]}]
gridsearch_model1=GridSearchCV(model1,model1_grid,cv=5)
gridsearch_model1.fit(reviews_train,labels_train)
print(gridsearch_model1.best_params_)

model2_grid=[{'n_estimators': [100, 200, 300],
'max_depth': [10,11,12,13,14,15],
'criterion': ['gini', 'entropy', 'log_loss']}]
gridsearch_model2=GridSearchCV(model2,model2_grid,cv=5)
gridsearch_model2.fit(reviews_train,labels_train)
print(gridsearch_model1.best_params_)

model3_grid=[{'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'adam']}]
gridsearch_model3=GridSearchCV(model1,model1_grid,cv=5)
gridsearch_model3.fit(reviews_train,labels_train)
print(gridsearch_model1.best_params_)

model4_grid=[{'C': [0.5, 1, 1.5, 2],
'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}]
gridsearch_model4=GridSearchCV(model4,model4_grid,cv=5)
gridsearch_model1.fit(reviews_train,labels_train)
print(gridsearch_model4.best_params_)

model5_grid=[{'n_estimators' : [50, 100, 150, 200],
'algorithm' : ['SAMME', 'SAMME.R']}]
gridsearch_model5=GridSearchCV(model5,model5_grid,cv=5)
gridsearch_model5.fit(reviews_train,labels_train)
print(gridsearch_model5.best_params_)

model1N=LogisticRegression(C=0.5, penalty='l2', random_state=1)
model2N=RandomForestClassifier(criterion='gini', max_depth=14, n_estimators=200, random_state=1)
model3N=MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(128, 128,), random_state=1)
model4N=SVC(C=2, kernel='rbf', probability=True, random_state=1)
model5N=AdaBoostClassifier(algorithm='SAMME.R', n_estimators=200, random_state=1)

VT=VotingClassifier(
    estimators=
    [
        ('lr', model1N),
        ('rf', model2N),
        ('ann', model3N),
        ('svc', model4N),
        ('abc', model5N)
    ],
     voting='soft'
)
VT.fit(counts_train,labels_train)
predicted=VT.predict(counts_test)
print(accuracy_score(predicted,labels_test))