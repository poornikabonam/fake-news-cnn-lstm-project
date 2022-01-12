from flask import Flask, render_template, request

import pickle

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Vocabulary Size
voc_size = 5000
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

nltk.download("stopwords")

ps = PorterStemmer()

def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

from keras.models import load_model
model = load_model('my_model.h5')

def predict(text):
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    print(corpus)
    one_hot_lbls = [one_hot(words,voc_size)for words in corpus]
    print(one_hot_lbls)
    Embedded_len = pad_sequences(one_hot_lbls,padding='pre',maxlen=20)
    print(Embedded_len)
    return model.predict_classes(Embedded_len)

def preprocess_news(news):
    p_news = re.sub('[^a-zA-Z]', ' ', news)
    p_news = p_news.lower().split()
    p_news = [ps.stem(word) for word in p_news if word not in stopwords.words('english')]
    p_news = ' '.join(p_news)
    return p_news

@app.route("/", methods = ['GET', 'POST'])
def home():
    fake_flag = False
    non_fake_flag = False
    danger = False
    message = ""
    try:
        if request.method == 'POST':
            dic = request.form.to_dict()
            news = dic['news']
            if len(news) == 0:
                raise Exception
            if predict(news)[0][0]== 1:
                fake_flag = True
                message = f"This NEWS is predicted as FAKE NEWS"
            else:
                non_fake_flag = True
                message = f"This NEWS is predicted as REAL NEWS"

    except Exception as e:
        print(e)
        danger = True
        message = "Please enter some text"
    return render_template("home.html", fake_flag = fake_flag, non_fake_flag = non_fake_flag, message = message, danger = danger)

if __name__ == '__main__':
    app.run(debug = True)