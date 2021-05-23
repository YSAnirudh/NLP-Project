import keras
import emoji
from keras.models import load_model
from keras.backend import concatenate, dtype
from tensorflow.core.protobuf.config_pb2 import OptimizerOptions
from tensorflow.keras.layers import Layer
from keras.layers.wrappers import Bidirectional
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Concatenate, Conv1D, Attention, GlobalAveragePooling1D
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from numpy import array
from numpy import asarray
from numpy import zeros
import pandas as pd
import numpy as np
import re
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.preprocessing import normalize
import difflib
from nltk.tokenize import word_tokenize
from sklearn.utils.extmath import weighted_mode
from tqdm import tqdm
import spacy

nlp = spacy.load('en_core_web_sm')



EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: \
:pensive: :ok_hand: :blush: :heart: :smirk: \
:grin: :notes: :flushed: :100: :sleeping: \
:relieved: :relaxed: :raised_hands: :two_hearts: :expressionless: \
:sweat_smile: :pray: :confused: :kissing_heart: :heartbeat: \
:neutral_face: :information_desk_person: :disappointed: :see_no_evil: :tired_face: \
:v: :sunglasses: :rage: :thumbsup: :cry: \
:sleepy: :yum: :triumph: :hand: :mask: \
:clap: :eyes: :gun: :persevere: :smiling_imp: \
:sweat: :broken_heart: :yellow_heart: :musical_note: :speak_no_evil: \
:wink: :skull: :confounded: :smile: :stuck_out_tongue_winking_eye: \
:angry: :no_good: :muscle: :facepunch: :purple_heart: \
:sparkling_heart: :blue_heart: :grimacing: :sparkles:".split(' ')
EMOJIS = [emoji for emoji in EMOJIS if emoji != '']




def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

def label_man(label_list, length):
    num_label_list = []
    for i in range(len(label_list)):
        label_list[i] = label_list[i].split(' ')
        temp = []
        for k in range(len(label_list[i])):
            for j in range(len(EMOJIS)):
                if(label_list[i][k] == EMOJIS[j]):
                    temp.append(j)
        num_label_list.append(temp)
        # print(smoothened)
    tempy = [0 for i in range(len(EMOJIS))]
    for i in range(len(num_label_list)):
        for j in range(len(num_label_list[i])):
            # print(num_label_list[i][j])
            tempy[num_label_list[i][j]] = tempy[num_label_list[i][j]] + 1

    # print(tempy)
    lis = []
    for i in range(len(EMOJIS)):
        lis.append([tempy[i], EMOJIS[i], i])

    lis.sort(reverse=True)
    label = []
    for i in range(len(EMOJIS)):
        if (i == length):
            break
        label.append(lis[i])
    emo = [[EMOJIS[i[2]], i[2]] for i in label]
    # print(label)
    num_label_list = []
    for i in range(len(label_list)):
        # label_list[i] = label_list[i].split(' ')
        temp = []
        for k in range(len(label_list[i])):
            for j in range(len(label)):
                if(label_list[i][k] == label[j][1]):
                    temp.append(j)
        num_label_list.append(temp)
    # print(num_label_list)
    # print(len(num_label_list))
    lebels = []
    for i in range(len(num_label_list)):
        tempy = [0 for i in range(len(label))]
        for j in range(len(num_label_list[i])):
            # print(num_label_list[i][j])
            tempy[num_label_list[i][j]] = tempy[num_label_list[i][j]] + 1
        lebels.append(tempy)
    return lebels, emo

max_amojis = 50
df = pd.read_csv('labels.csv')
df.drop('Unnamed: 0', axis = 1, inplace = True)
df = df[df['Emojis'].notna()]
df.reset_index(drop=True, inplace=True)
df['Tweet_Text'] = df['Tweet_Text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
# print(df)
tweet_list = df['Tweet_Text'].to_numpy().tolist()
label_list = df['Emojis'].to_numpy().tolist()

X = []
Y = []
toxic_comments = pd.read_csv('column_labels.csv')
sentences = list(tweet_list)
for sen in sentences:
    X.append(preprocess_text(sen))
# lbelis, emojis = label_man(label_list, max_amojis)
# print(emojis)
Y = toxic_comments.values
# print(Y.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 20

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


embeddings_dictionary = dict()

glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

deep_inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
LSTM_Layer_1 = LSTM(128)(embedding_layer)
dense_layer_1 = Dense(65, activation='sigmoid')(LSTM_Layer_1)
model = Model(inputs=deep_inputs, outputs=dense_layer_1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())
history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
