# Laden meiner Daten 
from LoadingAndPreprocessing.LoadingData import get_english_reviews
from LoadingAndPreprocessing.DataCleaning import *
from FeatureEngineering.FeatureEngineering import *
from ModelTraining.ModelTraining import *

from collections import Counter
from nltk.corpus import stopwords as sw
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from prettytable import PrettyTable
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_selection import chi2

#for ML MODELS
import sys
import numpy as np
import scipy as sp
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline



# Laden der Daten
path_to_csv = '/Users/milanagurbanova/Py_Stuff/MoviesReviews/IMDB-Dataset.csv'
data = get_english_reviews(path_to_csv)



## Globales
## Häufigkeitsanalyse ##Adjusting Stopwords
'''
data['Reviews'] = data['Reviews'].str.lower()
all_reviews = ' '.join(data['Reviews']) 
words = all_reviews.split()
word_counts = Counter(words)
#most_common_words = word_counts.most_common(100)
#print(most_common_words)'''

STOPWORDS = sw.words('english')
new_stopwords = ["the", "a", "of", "to", "is", "in", "i", "being", "made"]  # aus der Häufigkeitsanalyse ein paar
STOPWORDS.extend(new_stopwords)
STOPWORDS.remove("not")
STOPWORDS.remove("but")
STOPWORDS=set(STOPWORDS)



#######
####### DATA CLEANING 
#######

Data_cleaner = DataCleaner(STOPWORDS)
data['Reviews_clean'] = data['Reviews'].apply(Data_cleaner.data_cleaning)     #Data_cleaner.data_cleaning(data['Reviews'])
data['Reviews_clean'].head(10)

## TOKENISATION AND LEMMATIZATION 
Lemma_Tokenizer = LemmaTokenizer()
data['Reviews_clean'] = data['Reviews_clean'].apply(Lemma_Tokenizer.lemmame)

## DATA OVERVIEW /CHECKS
#  NA Values
print(data.isna().sum())

# Einblick in unsere Labels
print(data['Label'].describe())


## EXPLORATORY DATA ANALYSIS /LET'S SEE SOME GRAPHS!
# Check Distripution: von unseren Labels
sns.countplot(x=data['Label'])
plt.show()
print(data['Label'].value_counts())

# Check Distripution: charachters in reviews pro labelgruppe
character_distribution(data) 

# Check Distripution: words in reviews pro labelgruppe
word_distribution(data)

# Check Distripution: average length of reviews pro labelgruppe
average_review_length_distribution(data)




#######
####### DATA FEATURING
#######

## VECTORIZATION = WORD EMBEDDINGS (aber fortgeschrittener, Semantik wird einbezogen) 
                                  #(btw. advanced Alternative to the followoíng would be BERT /word2Vec / Glove)

## VECTORIZATION WITH UNI-, BI- and TRIGRAMS
train,test=train_test_split(data,test_size=.3,random_state=42, shuffle=True)#30%Test#aus skitlearn package
                                             #randome_state: stellt sicher, dass bei wdholung die selbe zufällige aufteilung der daten sattfindet -> wenn man 42 eingibt
                                             # die 42 ist also der "seed" für den zufallsgenerator (zahl selbst willkürlich gewählt)
countvect = CountVectorizer(analyzer = "word", tokenizer = LemmaTokenizer(), ngram_range=(1,3), min_df=10,max_features=5000)
tfidfvect = TfidfVectorizer(analyzer = "word", tokenizer = LemmaTokenizer(), ngram_range=(1,3),min_df=10,max_features=5000)
            # !!!  #wichtig dass tokenizer methode (LemmaTokenizer) gleich wie in vorverarbeitung: Konsistenz -> besonders da Vectorization der letzte Schirtt vor Modelltraining ist 
            # GEMEINSAMKEIT: #methods to convert text to numerical data; both from scikit-learn 
            # UNTERSCHIED DER VECTORIZER:
                    # CountVectorizer: zählt nur, wie oft ein Wort vorkommt
                    # Term Frequency - Inverse Document Frequency: oft vorkommende wörter -> leichter gewichtet und seltene schwerer

x_train_count = countvect.fit_transform(train['Reviews_clean']).toarray() #training/fit: identifizieren aller einzigartigen wörter und index zuordnung in einem vector -> transform: numerische vectoren unterschiedlicher länge, die der anz der wörter im vocanular entspricht
x_test_count = countvect.transform(test['Reviews_clean']).toarray()

x_train_tfidf = tfidfvect.fit_transform(train['Reviews_clean']).toarray()
x_test_tfidf = tfidfvect.transform(test['Reviews_clean']).toarray()

y_train = train['Label']
y_test = test['Label']

## FEATURE IMPORTANCE a) LOGISTISCHE REGRESSION AND COUNT VECTORIZER 
lgr = LogisticRegression()
lgr.fit(x_train_count,y_train)
lgr.score(x_test_count,y_test)  #accuracy meines modells
lgr.coef_[0]  #coefficients spiegeln wieder, wie sehr die entsprechenden features die zuordnung zu 1 oder 0 beeinflussen #Koeffizient von 0 keine Aussagekraft #KO positiv -> Kategorie 1
i=0
importantfeature = PrettyTable(["Feature", "Score"])
for feature, importance in zip(countvect.get_feature_names_out(), lgr.coef_[0]):
    if i<=200:
        importantfeature.add_row([feature, importance])
        i=i+1
print(importantfeature)

## FEATURE IMPORTANCE b) LOGISTISCHE REGRESSION AND TFIDF
lgr = LogisticRegression()
lgr.fit(x_train_tfidf,y_train)
lgr.score(x_test_tfidf,y_test)
lgr.coef_[0]
i=0
importantfeature = PrettyTable(["Feature", "Score"])
for feature, importance in zip(tfidfvect.get_feature_names_out(), lgr.coef_[0]):
    if i<=50:
        importantfeature.add_row([feature, importance])
        i=i+1
print(importantfeature)



## FEATURE SELECTION 
N = 5000
Number = 1
featureselection = PrettyTable(["Unigram", "Bigram","Trigram"]) # initialisiert ein PrettyTable-Objekt mit drei Spalten 

for category in train['Label'].unique():
    features_chi2 = chi2(x_train_tfidf, train['Label'] == category)
    # Chi-Quadrat-Test: Unabhängigkeit zw Features und Zielvariable -> starke Korrelationen erhalten höheren Chi-Quadrat-Score -> wir wollen hohe scores
    indices = np.argsort(features_chi2[0]) #sortieren nach diesen scores 
    feature_names = np.array(tfidfvect.get_feature_names_out())[indices]

    unigrams = [x for x in feature_names if len(x.split(' ')) == 1]
    bigrams = [x for x in feature_names if len(x.split(' ')) == 2]
    trigrams = [x for x in feature_names if len(x.split(' ')) == 3]

    print("%s. %s :" % (Number,category))
    print("\t# Unigrams :\n\t. %s" %('\n\t. '.join(unigrams[-N:])))
    print("\t# Bigrams :\n\t. %s" %('\n\t. '.join(bigrams[-N:])))
    print("\t# Trigrams :\n\t. %s" %('\n\t. '.join(trigrams[-N:])))
    Number += 1





#######
####### MODEL TRAINING
#######
    
## LOGISTISCHE REGRESSION
model_1 = LogisticRegression()

#Training
model_1.fit(x_train_tfidf,y_train)

#Evaluation
print("Precision Score on training dateset for Logistic Regression: %s" % precision_score(y_train,model_1.predict(x_train_tfidf),average='micro'))
print("AUC Score on training dateset for Logistic Regression: %s" % roc_auc_score(y_train,model_1.predict_proba(x_train_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_train_1 =f1_score(y_train,model_1.predict(x_train_tfidf),average="weighted")
print("F1 Score ftraining dateset for Logistic Regression: %s" % f1_score_train_1)
print("Precision Score on test for Logistic Regression: %s" % precision_score(y_test,model_1.predict(x_test_tfidf),average='micro'))
print("AUC Score on test for Logistic Regression: %s" % roc_auc_score(y_test,model_1.predict_proba(x_test_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_1 = f1_score(y_test,model_1.predict(x_test_tfidf),average="weighted")
print("F1 Score for Logistic Regression: %s" % f1_score_1)

'''
Precision Score on training dateset for Logistic Regression: 0.9185119047619048
AUC Score on training dateset for Logistic Regression: 0.9736989374675732
F1 Score ftraining dateset for Logistic Regression: 0.9185104707023269
Precision Score on test for Logistic Regression: 0.8963333333333333
AUC Score on test for Logistic Regression: 0.9606882858645622
F1 Score for Logistic Regression: 0.8963352656232666
Wall time: 6.47 s
'''


## Decision Tree Classifier
model_2 = Pipeline(
    steps=[
        #("classifier", DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)),
    ("classifier", DecisionTreeClassifier())
    ]
)

#Training
model_2.fit(x_train_tfidf,y_train)

#Evaluation
print("Precision Score on training dateset for Decision Tree Classifier: %s" % precision_score(y_train,model_2.predict(x_train_tfidf),average='micro'))
print("AUC Score on training dateset for Decision Tree Classifier: %s" % roc_auc_score(y_train,model_2.predict_proba(x_train_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_train_2 =f1_score(y_train,model_2.predict(x_train_tfidf),average="weighted")
print("F1 Score training dateset for Decision Tree Classifier: %s" % f1_score_train_2)
print("Precision Score on test for Decision Tree Classifier: %s" % precision_score(y_test,model_2.predict(x_test_tfidf),average='micro'))
print("AUC Score on test for Decision Tree Classifier: %s" % roc_auc_score(y_test,model_2.predict_proba(x_test_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_2 =f1_score(y_test,model_2.predict(x_test_tfidf),average="weighted")
print("F1 Score for Decision Tree Classifier: %s" % f1_score_2)



## Decision Tree Classifier with max depth 11 to fix overfit
model_3 = Pipeline(
    steps=[
        ("classifier", DecisionTreeClassifier( criterion='gini', max_depth=11, min_samples_split=2, min_samples_leaf=1)),
    ]
)

#Training
model_3.fit(x_train_tfidf,y_train)

#Evaluation
print("Precision Score on training dateset for Decision Tree Classifier: %s" % precision_score(y_train,model_3.predict(x_train_tfidf),average='micro'))
print("AUC Score on training dateset for Decision Tree Classifier: %s" % roc_auc_score(y_train,model_3.predict_proba(x_train_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_train_3 =f1_score(y_train,model_3.predict(x_train_tfidf),average="weighted")
print("F1 Score training dateset for Decision Tree Classifier: %s" % f1_score_train_3)
print("Precision Score on test for Decision Tree Classifier: %s" % precision_score(y_test,model_3.predict(x_test_tfidf),average='micro'))
print("AUC Score on test for Decision Tree Classifier: %s" % roc_auc_score(y_test,model_3.predict_proba(x_test_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_3 =f1_score(y_test,model_3.predict(x_test_tfidf),average="weighted")
print("F1 Score for Decision Tree Classifier: %s" % f1_score_3)



## Random Forest Classifier
model_4 = Pipeline(
    steps=[
        #("classifier", RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=15, min_samples_split=3, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None))
    ("classifier", RandomForestClassifier())]
)

#Training
model_4.fit(x_train_tfidf,y_train)

#Evaluation
print("Precision Score on training dateset for Random Forest Classifier: %s" % precision_score(y_train,model_4.predict(x_train_tfidf),average='micro'))
print("AUC Score on training dateset for Random Forest Classifier: %s" % roc_auc_score(y_train,model_4.predict_proba(x_train_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_train_4 =f1_score(y_train,model_4.predict(x_train_tfidf),average="weighted")
print("F1 Score training dateset for Random Forest Classifier: %s" % f1_score_train_4)
print("Precision Score on test for Random Forest Classifier: %s" % precision_score(y_test,model_4.predict(x_test_tfidf),average='micro'))
print("AUC Score on test for Random Forest Classifier: %s" % roc_auc_score(y_test,model_4.predict_proba(x_test_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_4 =f1_score(y_test,model_4.predict(x_test_tfidf),average="weighted")
print("F1 Score for Random Forest Classifier: %s" % f1_score_4)



## Ada Boost Classifier
model_5 = Pipeline(
    steps=[
        ("classifier", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=4),
    n_estimators=100,
    learning_rate=.8)),
    ]
)

#Training
model_5.fit(x_train_tfidf,y_train)

#Evaluation
print("Precision Score on training dateset for Ada Boost Classifier: %s" % precision_score(y_train,model_5.predict(x_train_tfidf),average='micro'))
print("AUC Score on training dateset for Ada Boost Classifier: %s" % roc_auc_score(y_train,model_5.predict_proba(x_train_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_train_5 =f1_score(y_train,model_5.predict(x_train_tfidf),average="weighted")
print("F1 Score training dateset for Ada Boost Classifier: %s" % f1_score_train_5)
print("Precision Score on test for Ada Boost Classifier: %s" % precision_score(y_test,model_5.predict(x_test_tfidf),average='micro'))
print("AUC Score on test for Ada Boost Classifier: %s" % roc_auc_score(y_test,model_5.predict_proba(x_test_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_5 =f1_score(y_test,model_5.predict(x_test_tfidf),average="weighted")
print("F1 Score for Random Forest Classifier: %s" % f1_score_5)



## HYPERPARAMETERTUENING

# LOGISTIC REGRESSION
param_gd={"penalty":["l2","l1"],
         "C":[0.01,0.1,1.0,10],
         "tol":[0.0001,0.001,0.01],
         "max_iter":[100,200]}
model_7, best_param = hyperparamtune(LogisticRegression(),param_gd,"accuracy",10,5)
# Evaluation
print("Precision Score on training dateset for Finetuned Logsitic Regression Classifier: %s" % precision_score(y_train,model_7.predict(x_train_tfidf),average='micro'))
print("AUC Score on training dateset for Finetuned Logsitic Regression Classifier: %s" % roc_auc_score(y_train,model_7.predict_proba(x_train_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_train_7 =f1_score(y_train,model_7.predict(x_train_tfidf),average="weighted")
print("F1 Score training dateset for Finetuned Logsitic Regression Classifier: %s" % f1_score_train_7)
print("Precision Score on test for Finetuned Logsitic Regression Classifier: %s" % precision_score(y_test,model_7.predict(x_test_tfidf),average='micro'))
print("AUC Score on test for Finetuned Logsitic Regression Classifier: %s" % roc_auc_score(y_test,model_7.predict_proba(x_test_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_7 =f1_score(y_test,model_7.predict(x_test_tfidf),average="weighted")
print("F1 Score for Finetuned Logsitic Regression Classifier: %s" % f1_score_7)


# RANDOM FOREST
param_gd={"n_estimators":[100,200,300],
         "max_depth":[11,13,17,19,23],
         "criterion":["gini","entropy"],
         "min_samples_split":[3,7,11],
         "min_samples_leaf":[3,5],
         "max_features":["sqrt", "log2"]}

model_8, best_param_8 = hyperparamtune(RandomForestClassifier(),param_gd,"accuracy",10,5)
#Evaluation
print("Precision Score on training dateset for Finetuned Random Forest Classifier: %s" % precision_score(y_train,model_8.predict(x_train_tfidf),average='micro'))
print("AUC Score on training dateset for Finetuned Random Forest Classifier: %s" % roc_auc_score(y_train,model_8.predict_proba(x_train_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_train_8 =f1_score(y_train,model_8.predict(x_train_tfidf),average="weighted")
print("F1 Score training dateset for Finetuned Random Forest Classifier: %s" % f1_score_train_8)
print("Precision Score on test for Finetuned Random Forest Classifier: %s" % precision_score(y_test,model_8.predict(x_test_tfidf),average='micro'))
print("AUC Score on test for Finetuned Random Forest Classifier: %s" % roc_auc_score(y_test,model_8.predict_proba(x_test_tfidf)[:,1],multi_class='ovo',average='macro'))
f1_score_8 =f1_score(y_test,model_8.predict(x_test_tfidf),average="weighted")
print("F1 Score for Finetuned Random Forest Classifier: %s" % f1_score_8)




###
### MODEL AUSWAHL
###
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


