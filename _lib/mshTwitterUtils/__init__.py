# custom libs
import karmahutils as kut
import karmahutils.batchutils as kutbatch
# nlp packages
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
# ML-NLP
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline
import tensorflow as tf

assert tf.__version__ >= "2.0"
from time import sleep
from geopy.geocoders import Nominatim

# model used initialized at start
tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine", use_fast=True)
model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


def get_geolocation_information(place_name, user_agent="datacrafting_exploTweets", geocoder=None, sleep_time=None):
    """ returns geolocation information using information from geopy
    for a lambda or batch use , it is recommanded to passe the geocoder defined outside the loop to reduce the network requests"""
    if place_name is None:
        return pd.Series({
            'address': '',
            'altitude': 0,
            'latitude': 0,
            'longitude': 0
        }
        )
    if geocoder is None:
        geocoder = Nominatim(user_agent=user_agent)
        location_info = geocoder.geocode(place_name)
    if sleep_time is not None:
        sleep(sleep_time)
    location_info_series = pd.Series({
        'address': location_info.address,
        'altitude': location_info.altitude,
        'latitude': location_info.latitude,
        'longitude': location_info.longitude
    }
    )
    return location_info_series


def add_location_info(
        data,
        location_name='place_name',
        user_agent="datacrafting_exploTweets",
        geocoder=None,
        sleep_time=None
):
    kut.display_message('adding geolocation information')
    start = kut.yet()
    if geocoder is None:
        geocoder = Nominatim(user_agent=user_agent)
    data_out = data.merge(
        data.apply(
            lambda row: get_geolocation_information(
                place_name=row[location_name],
                user_agent=user_agent,
                geocoder=geocoder,
                sleep_time=sleep_time
            ),
            axis=1
        ),
        left_index=True,
        right_index=True
    )
    kut.job_done(start=start)
    return data_out


def load_join_data(
        first_extract='mediaVaccination',
        second_extract='dataVaccination5G',
        format='dss',
        join_key='status_id',
        overlap='id'
):
    media_df = kut.load_dataset(first_extract) if not format == "csv" else pd.read_csv(first_extract)
    media_df = media_df.drop(overlap, axis=1).set_index(join_key)
    print('media df shape', media_df.shape)
    data_df = kut.load_dataset(second_extract) if not format == "csv" else pd.read_csv(second_extract)
    data_df.set_index(join_key, inplace=True)
    print('data df shape', data_df.shape)
    original_size = len(data_df)
    print('joining')
    data_df = data_df.join(media_df).reset_index()
    print(len(data_df), len(data_df) == original_size)
    print(data_df.shape)
    return data_df


def lemmatized_cleaning(text):
    tokenized = word_tokenize(text)  # Tokenize
    words_only = [word for word in tokenized if word.isalpha()]  # Remove numbers
    stop_words = set(stopwords.words('french'))  # Make stopword list
    without_stopwords = [word for word in words_only if not word in stop_words]  # Remove Stop Words
    lemma = WordNetLemmatizer()  # Initiate Lemmatizer
    lemmatized = [lemma.lemmatize(word) for word in without_stopwords]  # Lemmatize
    return ' '.join([str(X) for X in lemmatized])


def clean_tweets(data, tweet_column='tweet', method=lemmatized_cleaning, clean_column=None):
    kut.display_message('cleaning tweet column', 'name: ' + tweet_column)
    if clean_column is None:
        clean_column = tweet_column
    print('output', clean_column)
    start = kut.yet()
    # Apply to all texts
    data[clean_column] = data[tweet_column].apply(lambda x: method(x))
    kut.job_done(start=start)
    return data


def add_sentiment_information(data, lemme_column='lemmatizedTweet'):
    kut.display_message('adding sentiment information based on ' + lemme_column)
    print('expected time is 300s/1000rows')
    start = kut.yet()
    lemme_array = data[lemme_column].tolist()
    nlp_df = pd.DataFrame(nlp(lemme_array)).rename(columns={'label': 'sentiment_label', 'score': 'sentiment_score'})
    out = data.join(pd.DataFrame(nlp(lemme_array)))
    kut.job_done(start=start)
    return out
