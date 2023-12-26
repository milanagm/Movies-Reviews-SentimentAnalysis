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
 

## LEMMATIZIEREN UND TOKENIZER KLASSE ##
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def lemmamee(self, reviews):
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]


