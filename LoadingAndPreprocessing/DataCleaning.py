# Laden meiner Daten 
from LoadingData import get_english_reviews

# imports 
from nltk.corpus import stopwords as sw #standard stopwords 
from collections import Counter
import pandas as pd
import re #regular expressions (for strings) e.g. substituieren mit re.sub
from sklearn.base import BaseEstimator, TransformerMixin 
from nltk import word_tokenize    
from nltk.stem import WordNetLemmatizer # e.g. playing -> play  
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



## Special Characters Removal/ Transformation: definieren von funktionen
# special character
def remove_special_character(content):
    return re.sub('\W+',' ', content )  #re.sub('\[[^&@#!]]*\]', '', content)

# URL's
def remove_url(content):
    return re.sub(r'http\S+', '', content)

# stopwords
def remove_stopwords(content, stop_words):
    clean_data = []
    for i in content.split():
        if i.strip().lower() not in stop_words and i.strip().lower().isalpha():
            clean_data.append(i.strip().lower())
    return " ".join(clean_data)

# Transformation
def contraction_expansion(content):
    content = re.sub(r"won\'t", "would not", content) #regex = r"won\'t"
    content = re.sub(r"can\'t", "can not", content)
    content = re.sub(r"don\'t", "do not", content)
    content = re.sub(r"shouldn\'t", "should not", content)
    content = re.sub(r"needn\'t", "need not", content)
    content = re.sub(r"hasn\'t", "has not", content)
    content = re.sub(r"haven\'t", "have not", content)
    content = re.sub(r"weren\'t", "were not", content)
    content = re.sub(r"mightn\'t", "might not", content)
    content = re.sub(r"didn\'t", "did not", content)
    content = re.sub(r"n\'t", " not", content)
    return content


## EIGENTLICHE DATENBEREINIGUNG ## METHODEN ZUR AUSFÜHRUNG
## Data Cleaning ##
class DataCleaner():
    def __init__(self, stopwords):
        self.stop_words = stopwords

    def data_cleaning(self, content):
        content = contraction_expansion(content)
        content = remove_special_character(content)
        content = remove_url(content)
        content = remove_stopwords(content, self.stop_words)      
        return content
 




## EXPLORATORY DATA ANALYSIS /LET'S SEE SOME GRAPHS ##
# Check Distripution: charachters in reviews pro labelgruppe
def character_distribution(data):
    figure,(pos_ax,neg_ax)=plt.subplots(1,2,figsize=(15,8))
    figure.suptitle('Number of Characters in reviews')

    len_pos_review=data[data['Ratings']>=7]['Reviews_clean'].str.len()
    pos_ax.hist(len_pos_review,color='green')
    pos_ax.set_title('Positive Reviews')

    len_neg_review=data[data['Ratings']<=4]['Reviews_clean'].str.len()
    neg_ax.hist(len_neg_review,color='red')
    neg_ax.set_title('Negative Reviews')
    plt.show()

# Check Distripution: words in reviews pro labelgruppe
def word_distribution(data):    
    figure,(pos_ax,neg_ax)=plt.subplots(1,2,figsize=(15,8))
    figure.suptitle('Number of Words')

    pos_word=data[data['Ratings']>=7]['Reviews_clean'].str.split().map(lambda review: len(review))
    pos_ax.hist(pos_word,color='green')
    pos_ax.set_title('Number of Words in Positive Reviews')

    neg_word=data[data['Ratings']<=4]['Reviews_clean'].str.split().map(lambda review: len(review))
    neg_ax.hist(neg_word,color='red')
    neg_ax.set_title('Number of Words Negative Reviews')
    plt.show()

# Check Distripution: average length of reviews pro labelgruppe
def average_review_length_distribution(data):    
    figure,(pos_ax,neg_ax)=plt.subplots(1,2,figsize=(15,8))
    figure.suptitle('Average word length in Reviews')

    pos_word=data[data['Ratings']>=7]['Reviews_clean'].str.split().apply(lambda x : [len(i) for i in x])
    sns.distplot(pos_word.map(lambda x: np.mean(x)),ax=pos_ax,color='green')
    pos_ax.set_title('Positive Reviews')

    neg_word=data[data['Ratings']<=4]['Reviews_clean'].str.split().apply(lambda x : [len(i) for i in x])
    sns.distplot(neg_word.map(lambda x: np.mean(x)),ax=neg_ax,color='red')
    neg_ax.set_title('Negative Reviews')