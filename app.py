import nltk
import re
from bert import bert_tokenization
import tensorflow_hub as hub
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template("home.html")


@app.route('/', methods=['POST'])
def classify():
    global classification_model
    global tokenizer

    post = request.form['post']
    post = post_tokenizing_dataset3(post)
    tokenized_post = tokenizer.tokenize(post)
    tokenized_sequence = [tokenizer.convert_tokens_to_ids(tokenized_post)]
    test_padded = pad_sequences(
        tokenized_sequence, maxlen=10, padding="post", truncating="post"
    )
    predicted_value = classification_model.predict(test_padded)
    if predicted_value[0]>0.7:
        result="The tweet has CyberBullying content !!!"
    else:
        result = "The tweet is fine to Post !!!"

    return render_template("home.html", tweet=result)


def post_tokenizing_dataset3(text):

    wpt = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')
    lancaster_stemmer = nltk.stem.lancaster.LancasterStemmer()
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()

    filter_val = []
    value = re.sub(r'@\w*', '', text)
    value = re.sub(r'&.*;', '', value)
    value = re.sub(r'http[s?]?:\/\/.*[\r\n]*', '', value)
    tokens = [token for token in wpt.tokenize(value)]
    tokens = [word for word in tokens if word.isalpha()]
    lemma_tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) >= 3]
    lancaster_tokens = [lancaster_stemmer.stem(word) for word in lemma_tokens]
    if len(lancaster_tokens) != 0:
        return ' '.join(lancaster_tokens)


if __name__ == '__main__':
    classification_model = load_model('CNN_MODEL_FOR_POF.PKL')
    BertTokenizer = bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=False)
    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
    app.run()


