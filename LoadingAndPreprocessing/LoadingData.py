# imports 
import pandas as pd 
from langdetect import detect



## DATEN LADEN ##
# damit ich von einer anderen File auf den neuen "englischen" Datensatz zugreifen kann
def load_dataset(path):
    df = pd.read_csv(path, encoding='latin-1')
    df['Label'] = df['Ratings'].apply(lambda x: 1 if x >= 7 else (0 if x <= 4 else 2))  #Featureengineering
    df = df[df.Label < 2]
    data = df[['Reviews', 'Label']]
    return data

def get_english_reviews(path_to_csv):
    data = load_dataset(path_to_csv)
    print(data.keys())

    ## NUR ENGLISCHE DATEN BEHALTEN ##dauert schon mind. 5min
    unique_languages = set()      # set für die entdeckten sprachen im text 
    for text in data['Reviews']:  # Spaltenname mit dem relevanten Text
        try:
            # Erkennen der Sprache meiner Daten
            language = detect(text)
            # Hinzufügen der Sprache zu meinem set unique_languages
            unique_languages.add(language)
        except Exception as e:
            # Wenn ein Fehler bei der Spracherkennung auftritt
            print(f"Error detecting language: {e}")

    # ich füge eine Spalte language hinzu, um dann alles was nicht englisch ist auszuschließen
    data['language'] = data['Reviews'].apply(lambda text: detect(text) if text.strip() != "" else "")
    # exkludieren der nicht englischen reviews 
    data = data[data['language'] == 'en'] 
    #rausnehmen der spalte englisch
    data = data[['Reviews', 'Label']]

    return data

'''
## btw: folgende Sprachen gabs:
for text in data['Reviews']:  # Spaltenname mit dem relevanten Text
    try:
        # Erkennen der Sprache meiner Daten
        language = detect(text)
        # Hinzufügen der Sprache zu meinem set unique_languages
        unique_languages.add(language)
    except Exception as e:
        # Wenn ein Fehler bei der Spracherkennung auftritt
        print(f"Error detecting language: {e}")

print(unique_languages) 
#Ausgabe war: {'pl', 'hu', 'en', 'no', 'hr', 'pt', 'id', 'es', 'fr', 'it', 'cy', 'sv', 'so', 'tl', 'fi', 'tr', 'de', 'af', 'nl', 'da'}
'''


