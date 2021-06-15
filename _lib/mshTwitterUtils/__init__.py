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

# model used initialized at start
tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine", use_fast=True)
model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


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


def add_sentiment_information(data,lemme_column='lemmatizedTweet'):
    kut.display_message('adding sentiment information based on '+ lemme_column)
    print('expected time is 300s/1000rows')
    start=kut.yet()
    lemme_array=data[lemme_column].tolist()
    nlp_df=pd.DataFrame(nlp(lemme_array)).rename(columns={'label':'sentiment_label','score':'sentiment_score'})
    out=data.join(pd.DataFrame(nlp(lemme_array)))
    kut.job_done(start=start)
    return out