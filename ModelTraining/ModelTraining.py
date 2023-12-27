import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection

import sys
import scipy as sp
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline




## MODEL EVALUATION
## LOGISTISCHE REGRESSION
def evaluation_log_reg(x_train_tfidf, x_test_tfidf, y_train, y_test, model_1):
    print("Precision Score on training dateset for Logistic Regression: %s" % precision_score(y_train,model_1.predict(x_train_tfidf),average='micro'))
    print("AUC Score on training dateset for Logistic Regression: %s" % roc_auc_score(y_train,model_1.predict_proba(x_train_tfidf)[:,1],multi_class='ovo',average='macro'))
    f1_score_train_1 =f1_score(y_train,model_1.predict(x_train_tfidf),average="weighted")
    print("F1 Score ftraining dateset for Logistic Regression: %s" % f1_score_train_1)
    print("Precision Score on test for Logistic Regression: %s" % precision_score(y_test,model_1.predict(x_test_tfidf),average='micro'))
    print("AUC Score on test for Logistic Regression: %s" % roc_auc_score(y_test,model_1.predict_proba(x_test_tfidf)[:,1],multi_class='ovo',average='macro'))
    f1_score_1 = f1_score(y_test,model_1.predict(x_test_tfidf),average="weighted")
    print("F1 Score for Logistic Regression: %s" % f1_score_1)
    return f1_score_train_1, f1_score_1


## Decision Tree Classifier
def evaluation_decision_tree(x_train_tfidf, x_test_tfidf, y_train, y_test, model_2):
    print("Precision Score on training dateset for Decision Tree Classifier: %s" % precision_score(y_train,model_2.predict(x_train_tfidf),average='micro'))
    print("AUC Score on training dateset for Decision Tree Classifier: %s" % roc_auc_score(y_train,model_2.predict_proba(x_train_tfidf)[:,1],multi_class='ovo',average='macro'))
    f1_score_train_2 =f1_score(y_train,model_2.predict(x_train_tfidf),average="weighted")
    print("F1 Score training dateset for Decision Tree Classifier: %s" % f1_score_train_2)
    print("Precision Score on test for Decision Tree Classifier: %s" % precision_score(y_test,model_2.predict(x_test_tfidf),average='micro'))
    print("AUC Score on test for Decision Tree Classifier: %s" % roc_auc_score(y_test,model_2.predict_proba(x_test_tfidf)[:,1],multi_class='ovo',average='macro'))
    f1_score_2 =f1_score(y_test,model_2.predict(x_test_tfidf),average="weighted")
    print("F1 Score for Decision Tree Classifier: %s" % f1_score_2)
    return f1_score_train_2, f1_score_2



## Decision Tree Classifier with max depth 11 to fix overfit
def evaluation_decision_tree_maxdepth(x_train_tfidf, x_test_tfidf, y_train, y_test, model_3):
    print("Precision Score on training dateset for Decision Tree Classifier: %s" % precision_score(y_train,model_3.predict(x_train_tfidf),average='micro'))
    print("AUC Score on training dateset for Decision Tree Classifier: %s" % roc_auc_score(y_train,model_3.predict_proba(x_train_tfidf)[:,1],multi_class='ovo',average='macro'))
    f1_score_train_3 =f1_score(y_train,model_3.predict(x_train_tfidf),average="weighted")
    print("F1 Score training dateset for Decision Tree Classifier: %s" % f1_score_train_3)
    print("Precision Score on test for Decision Tree Classifier: %s" % precision_score(y_test,model_3.predict(x_test_tfidf),average='micro'))
    print("AUC Score on test for Decision Tree Classifier: %s" % roc_auc_score(y_test,model_3.predict_proba(x_test_tfidf)[:,1],multi_class='ovo',average='macro'))
    f1_score_3 =f1_score(y_test,model_3.predict(x_test_tfidf),average="weighted")
    print("F1 Score for Decision Tree Classifier: %s" % f1_score_3)
    return f1_score_train_3, f1_score_3



## Random Forest Classifier
def evaluation_random_forest(x_train_tfidf, x_test_tfidf, y_train, y_test, model_4):
    print("Precision Score on training dateset for Random Forest Classifier: %s" % precision_score(y_train,model_4.predict(x_train_tfidf),average='micro'))
    print("AUC Score on training dateset for Random Forest Classifier: %s" % roc_auc_score(y_train,model_4.predict_proba(x_train_tfidf)[:,1],multi_class='ovo',average='macro'))
    f1_score_train_4 =f1_score(y_train,model_4.predict(x_train_tfidf),average="weighted")
    print("F1 Score training dateset for Random Forest Classifier: %s" % f1_score_train_4)
    print("Precision Score on test for Random Forest Classifier: %s" % precision_score(y_test,model_4.predict(x_test_tfidf),average='micro'))
    print("AUC Score on test for Random Forest Classifier: %s" % roc_auc_score(y_test,model_4.predict_proba(x_test_tfidf)[:,1],multi_class='ovo',average='macro'))
    f1_score_4 =f1_score(y_test,model_4.predict(x_test_tfidf),average="weighted")
    print("F1 Score for Random Forest Classifier: %s" % f1_score_4)
    return f1_score_train_4, f1_score_4



## Ada Boost Classifier
def evaluation_ada_boost(x_train_tfidf, x_test_tfidf, y_train, y_test, model_5):
    print("Precision Score on training dateset for Ada Boost Classifier: %s" % precision_score(y_train,model_5.predict(x_train_tfidf),average='micro'))
    print("AUC Score on training dateset for Ada Boost Classifier: %s" % roc_auc_score(y_train,model_5.predict_proba(x_train_tfidf)[:,1],multi_class='ovo',average='macro'))
    f1_score_train_5 =f1_score(y_train,model_5.predict(x_train_tfidf),average="weighted")
    print("F1 Score training dateset for Ada Boost Classifier: %s" % f1_score_train_5)
    print("Precision Score on test for Ada Boost Classifier: %s" % precision_score(y_test,model_5.predict(x_test_tfidf),average='micro'))
    print("AUC Score on test for Ada Boost Classifier: %s" % roc_auc_score(y_test,model_5.predict_proba(x_test_tfidf)[:,1],multi_class='ovo',average='macro'))
    f1_score_5 =f1_score(y_test,model_5.predict(x_test_tfidf),average="weighted")
    print("F1 Score for Random Forest Classifier: %s" % f1_score_5)
    return f1_score_train_5, f1_score_5





## HYPERPARAMETER TUENING

def hyperparamtune(classifier, param_grid, metric, verbose_value,cv, x_train_tfidf, y_train):
    model=model_selection.GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            scoring=metric,
            verbose=verbose_value,            
            cv=cv)

    model.fit(x_train_tfidf,y_train)
    print("Best Score %s" % {model.best_score_})
    print("Best hyperparameter set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
    return model, best_parameters





## MODEL AUSWAHL
def modelauswahl (f1_score_1, f1_score_2, f1_score_3, f1_score_4, f1_score_5):
    best_score = 0
    model = None
    if ((f1_score_1>f1_score_2) & (f1_score_1>f1_score_3) & (f1_score_1>f1_score_4) & (f1_score_1>f1_score_5)& (f1_score_1>f1_score_7)&(f1_score_1>f1_score_8)):
        model = model_1
        print("Logsitics Regression is providing best F1 score: %f" % f1_score_1)
    elif((f1_score_2>f1_score_1) & (f1_score_2>f1_score_3) & (f1_score_2>f1_score_4) & (f1_score_2>f1_score_5)& (f1_score_2>f1_score_7)&(f1_score_2>f1_score_8)):
        model = model_2
        print(" Over fit Decision Tree is providing best F1 score: %f" % f1_score_2)
    elif((f1_score_3>f1_score_1) & (f1_score_3>f1_score_2) & (f1_score_3>f1_score_4)&(f1_score_3>f1_score_5)& (f1_score_3>f1_score_7)&(f1_score_3>f1_score_8)):
        model = model_3
        print("Decision Tree is providing best F1 score: %f" % f1_score_3)
    elif((f1_score_4>f1_score_1) & (f1_score_4>f1_score_2) & (f1_score_4>f1_score_3)&(f1_score_4>f1_score_5)& (f1_score_4>f1_score_7)&(f1_score_4>f1_score_8)):
        model = model_4
        print("Random Forest is providing best F1 score: %f" % f1_score_4)
    elif((f1_score_5>f1_score_1) & (f1_score_5>f1_score_2) & (f1_score_5>f1_score_4)&(f1_score_5>f1_score_3)& (f1_score_5>f1_score_7)&(f1_score_5>f1_score_8)):
        model = model_5
        print("Adaboost Classifier is providing best F1 score: %f" % f1_score_5)
    elif((f1_score_7>f1_score_1) & (f1_score_7>f1_score_2) & (f1_score_7>f1_score_4)&(f1_score_7>f1_score_3)& (f1_score_7>f1_score_5)&(f1_score_7>f1_score_8)):
        model = model_7
        print("Finetuned Logsitics Regression Classifier is providing best F1 score: %f" % f1_score_7)
    elif((f1_score_8>f1_score_1) & (f1_score_8>f1_score_2) & (f1_score_8>f1_score_4)&(f1_score_8>f1_score_3)& (f1_score_8>f1_score_7)&(f1_score_8>f1_score_5)):
        model = model_8
        print("Finetuned Random Forest Classifier is providing best F1 score: %f" % f1_score_8)
    else:
        print("No Model is selected, Train again")