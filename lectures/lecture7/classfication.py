from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV

#read the reviews and their polarities from a given file
def loadData(fname):
    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:
        review,rating=line.strip().split('\t')
        reviews.append(review.lower())
        labels.append(int(rating))
    f.close()
    return reviews,labels

rev_train,labels_train=loadData('reviews_train.txt')
rev_test,labels_test=loadData('reviews_test.txt')

## TF-IDF
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 3))
train_tfidf = tfidf.fit_transform(rev_train)
test_tfidf = tfidf.transform(rev_test)

#Build a counter based on the training dataset
counter = CountVectorizer()
counter.fit(rev_train)


#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(rev_train)#transform the training data
counts_test = counter.transform(rev_test)#transform the testing data

#train classifier
model1 = SGDClassifier()
model2 = RandomForestClassifier()
model3 = XGBClassifier()
model4 = MLPClassifier()
model5 = SVC()

predictors=[('SGD',model1),('RandomForest',model2),('XGBoost',model3),('MLP',model4),('SVM',model5)]

VT=VotingClassifier(predictors)

#train all classifier on the same datasets
VT.fit(counts_train,labels_train)

#use hard voting to predict (majority voting)
pred_countvec=VT.predict(counts_test)

#print accuracy
print (accuracy_score(pred_countvec,labels_test))

#train all classifier on the same datasets
VT.fit(train_tfidf,labels_train)

#use hard voting to predict (majority voting)
pred_tfidf=VT.predict(test_tfidf)

#print accuracy
print (accuracy_score(pred_tfidf,labels_test))

#=======================================================================================
## SGD
#build the parameter grid

SGD_grid = [{'loss':['hinge', 'log'],
             'alpha':[0.0001,0.01,0.1],
             'learning_rate' : ['optimal','adaptive']
            }]


#build a grid search to find the best parameters
gridsearchSGD = GridSearchCV(model1, SGD_grid, cv=5)

#run the grid search
gridsearchSGD.fit(train_tfidf,labels_train)

#=======================================================================================
## Random Forest
#build the parameter grid

RF_grid = [{'n_estimators' : [100,300,500,600],
            'criterion' : ['gini', 'entropy'],
            'random_state' : [12, 26, 44]
           }]

#build a grid search to find the best parameters
gridsearchRF  = GridSearchCV(model2, RF_grid, cv=5)

#run the grid search
gridsearchRF.fit(train_tfidf,labels_train)

#=======================================================================================
## XGBoost
#build the parameter grid
XGB_grid = [{'booster' : ['gbtree', 'gblinear'],
            'max_depth' : [10,22],
             'n_estimators' : [100,200]
            }]


#build a grid search to find the best parameters
gridsearchXGB  = GridSearchCV(model3, XGB_grid, cv=5)

#run the grid search
gridsearchXGB.fit(train_tfidf,labels_train)

#=======================================================================================
## Multi-Layer Perceptron
#build the parameter grid
MLP_grid = [{'hidden_layer_sizes' : [(100,), (50,40,30,20,20,10)],
            'solver' : ['adam' , 'sgd'],
             'activation' : ['relu','logistic'],
             'max_iter' : [200,500]
            }]

#build a grid search to find the best parameters
gridsearchMLP  = GridSearchCV(model4, MLP_grid, cv=5)

#run the grid search
gridsearchMLP.fit(train_tfidf,labels_train)

#=======================================================================================
## SVM
#build the parameter grid
SVM_grid = [{'kernel' : ['rbf', 'sigmoid'],
            'gamma' : ['scale','auto'],
             'random_state' : [10,20,15]
            }]

#build a grid search to find the best parameters
gridsearchSVM  = GridSearchCV(model5, SVM_grid, cv=5)

#run the grid search
gridsearchSVM.fit(train_tfidf,labels_train)


#=======================================================================================
def show_best(gridsearchModel):
    for param_name in gridsearchModel.best_params_:
        print(param_name, gridsearchModel.best_params_[param_name])

show_best(gridsearchSGD)

show_best(gridsearchRF)

show_best(gridsearchXGB)

show_best(gridsearchMLP)

show_best(gridsearchSVM)

### Using the best parameter for each model
model1 = SGDClassifier(alpha = 0.0001,learning_rate = 'optimal', loss ='hinge')
model2 = RandomForestClassifier(criterion = 'entropy',n_estimators =300, random_state =12 )
model3 = XGBClassifier(booster ='gbtree',max_depth =22,n_estimators =200  )
model4 = MLPClassifier(activation ='relu', hidden_layer_sizes =(100,), max_iter =200, solver ='adam'  )
model5 = SVC(gamma ='scale',kernel ='sigmoid',random_state =10 )

predictors=[('SGD',model1),('RandomForest',model2),('XGBoost',model3),('MLP',model4),('SVM',model5)]

VT=VotingClassifier(predictors)

VT.fit(train_tfidf,labels_train)

#use the VT classifier to predict
predicted=VT.predict(test_tfidf)

#print the accuracy
print (accuracy_score(predicted,labels_test))

#print the accuracy
print ('Using the best parameters of GridSearch, we get an accuracy of :',accuracy_score(predicted,labels_test))