# Laden meiner Daten 
from LoadingData import get_english_reviews
from collections import Counter
from nltk.corpus import stopwords as sw
from DataCleaning import DataCleaner, LemmaTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


## AUSFÜHRUNG
# Laden der Daten
path_to_csv = '/Users/milanagurbanova/Py_Stuff/MoviesReviews/IMDB-Dataset.csv'
data = get_english_reviews(path_to_csv)


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


## DATA CLEANING 
Data_cleaner = DataCleaner(STOPWORDS)
data['Reviews'] = data['Reviews'].apply(Data_cleaner.data_cleaning)     #Data_cleaner.data_cleaning(data['Reviews'])
data['Reviews'].head(10)


### FEATURE ENGINEERING ###
## TOKENISATION AND LEMMATIZATION 
Lemma_Tokenizer = LemmaTokenizer()
data['Reviews'] = data['Reviews'].apply(Lemma_Tokenizer.lemmamee)


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

# Check Distripution: words in reviews pro labelgruppe

# Check Distripution: average length of reviews pro labelgruppe
    