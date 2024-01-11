from vnlp import StemmerAnalyzer
from vnlp import StopwordRemover
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize, sent_tokenize
import math
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences

stp_rmv = StopwordRemover()
stemmer = StemmerAnalyzer()

#CHROMADB RELATED FUNCTIONS

def combine(sentence):
    whole = ''
    for i in sentence:
        whole += i+' '
    return whole

def stem_stop_remover(sentence):

    temp = stemmer.predict(str(sentence))
    final = stp_rmv.drop_stop_words(temp)
    return final

def collection_get(name):
    client = chromadb.PersistentClient(path="/nlp_pr")
    collection = client.get_collection(name)
    return collection

def collection_create(name):
    trb_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
    client = chromadb.PersistentClient(path="/nlp_pr")
    collection = client.get_or_create_collection(name, embedding_function=trb_ef)
    return collection

def sorgula(sentence,model_name):
    upd_sent = stem_stop_remover(sentence)
    model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
    sample_embedding = model.encode(upd_sent)
    upd = sample_embedding.tolist()
    client = chromadb.PersistentClient(path="/nlp_pr")
    clt = client.get_collection(model_name)
    found = clt.query(query_embeddings=upd, n_results=10)
    return found

def get_answers(dbfound,sentences):
    f_keys = []
    for i in range(0, len(dbfound['ids'][0])):
        f_keys.append(int(dbfound['ids'][0][i]))

    text = ' '
    for i in range(0,len(f_keys)):
        text += sentences[f_keys[i]]+' '

    textp = combine(stem_stop_remover(text))

    return textp


#EXTRACTION RELATED FUNCTIONS

def tf_mat(sentences: list) -> dict:

    tf_matrix = {}

    for sentence in sentences:
        tf_table = {}

        clean_words = word_tokenize(sentence)
        words_count = len(word_tokenize(sentence))

        word_freq = {}
        for word in clean_words:
            word_freq[word] = (word_freq[word] + 1) if word in word_freq else 1

        for word, count in word_freq.items():
            tf_table[word] = count / words_count

        tf_matrix[sentence[:15]] = tf_table

    return tf_matrix

def idf_mat(sentences: list) -> dict:

    idf_matrix = {}
    documents_count = len(sentences)
    sentence_word_table = {}

    for sentence in sentences:
        clean_words = word_tokenize(sentence)
        sentence_word_table[sentence[:15]] = clean_words

    word_in_docs = {}
    for sent, words in sentence_word_table.items():
        for word in words:
            word_in_docs[word] = (word_in_docs[word] + 1) if word in word_in_docs else 1

    for sent, words in sentence_word_table.items():
        idf_table = {}
        for word in words:
            idf_table[word] = math.log10(documents_count / float(word_in_docs[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

def tf_idf_mat(tf_matrix, idf_matrix) -> dict:

    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

def score_table(tf_idf_matrix) -> dict:

    sentence_value = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        smoothing = 1
        sentence_value[sent] = (total_score_per_sentence + smoothing) / (count_words_in_sentence + smoothing)


    return sentence_value

def extract_sum(sentences, sentence_value):
    sum = 0
    for val in sentence_value:
        sum += sentence_value[val]

    threshold = sum / len(sentence_value)

    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentence_value and sentence_value[sentence[:15]] >= threshold:
            summary += sentence + " "
            sentence_count += 1

    return summary

def extraction(text):

    sentences = sent_tokenize(text)

    tf_matrix = tf_mat(sentences)

    idf_matrix = idf_mat(sentences)

    tf_idf_matrix = tf_idf_mat(tf_matrix, idf_matrix)

    sentence_value = score_table(tf_idf_matrix)

    summary = extract_sum(sentences, sentence_value)

    return summary

#RNN RELATED FUNCTIONS

def formation(text):
    uq_tokenized = word_tokenize(text)

    # organize into sequences of tokens
    length = 10 + 1  #cümleler 10+1 kelimeden oluştuğu varsayılmakta
    sequences = list()

    for i in range(length, len(uq_tokenized)):
        # select sequence of tokens
        temp = uq_tokenized[i - length:i]
        # convert into a line
        line = ' '.join(temp)
        # store
        sequences.append(line)

    return sequences

def encoding(sequences):
    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    en_sequences = tokenizer.texts_to_sequences(sequences)
    vocab_size = len(tokenizer.word_index) + 1

    return en_sequences,vocab_size,tokenizer

#in main vocab_size = len(tokenizer.word_index) + 1

def in_x(en_sequences):
    x_en_sequences = np.array(en_sequences)[:, :-1]
    return x_en_sequences

#do in main seq_length = x.shape[1]

def categorized_y(en_sequences,vocab_size):
    y_en_sequences = np.array(en_sequences)[:, -1]
    y = to_categorical(y_en_sequences, num_classes=vocab_size)
    return y

def get_model(vocab_size,seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    # compile network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summarize defined model
    model.summary()
    return model

def train_model(model,x_in,cat):
    return model.fit(x_in,cat,batch_size = 128, epochs=20)


def generation(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = stem_stop_remover(seed_text)
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integers
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')  #difference from the origianl code that belongs to the book
        # predict probabilities for each word
        yhat_probs = model.predict(encoded, verbose=0)
        # get the word index with the highest probability
        yhat = np.argmax(yhat_probs)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

