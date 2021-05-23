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
sentences = list(tweet_list)
for sen in sentences:
    X.append(preprocess_text(sen))
lbelis, emojis = label_man(label_list, max_amojis)
print(emojis)
Y = np.asarray(lbelis, dtype='int64')
# print(Y.tolist())


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
# X_train.append("Do you agree with Elon?")

# print(y_train)

ent = pd.read_csv('mainkg/embeddings/transe/ent_embedding.tsv', sep="\t", header=None)
entity = ent.to_numpy()
# print(ent)
rel = pd.read_csv('mainkg/embeddings/transe/rel_embedding.tsv', sep="\t", header=None)
relation = rel.to_numpy()
# print(rel)

ent_lab = pd.read_csv('mainkg/embeddings/transe/correct_ent_labels.csv')
ent_lab.drop('Unnamed: 0', axis = 1, inplace = True)
enlab = ent_lab.to_numpy()
rel_lab = pd.read_csv('mainkg/embeddings/transe/correct_rel_labels.csv')
rel_lab.drop('Unnamed: 0', axis = 1, inplace = True)
relab = rel_lab.to_numpy()
rels  = []
for i in relab:
    rels.append(i[0])
# cosine_similarity()
# print(rels)

kgwords = []
for i in enlab:
    kgwords.append(str(i[0]))
for i in relab:
    kgwords.append(str(i[0]))


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# KG_train, KG_test = train_test_split(kgwords, test_size=0.20, random_state=42)

kgword_dict = dict()

for i in range(len(kgwords)):
    kgword_dict[kgwords[i]] = i

KG_train = [list() for i in range(len(X_train))]
KG_test = [list() for i in range(len(X_test))]

# print(kgword_dict)
for i in tqdm(range(len(X_train))):
    listy = word_tokenize(X_train[i].strip())
    # print(len(listy))
    for j in range(len(listy)):
        # x = difflib.get_close_matches(listy[j], kgwords, n=1, cutoff=0.7)
        
        # for k in kgword_dict:
        if listy[j] in kgword_dict:
            KG_train[i].append(kgword_dict[listy[j]])
# elon musk 
        # x = difflib.get_close_matches(listy[j], kgwords, n=1, cutoff=0.9)
        # if listy[j] in kgword_dict:
        # if (len(x) != 0):
        #     KG_train[i].append(x[0])
for i in tqdm(range(len(X_test))):
    listy = word_tokenize(X_test[i].strip())
    # print(len(listy))
    for j in range(len(listy)):
        if listy[j] in kgword_dict:
            KG_test[i].append(kgword_dict[listy[j]])
        # for k in kgword_dict:
        # x = difflib.get_close_matches(listy[j], kgwords, n=1, cutoff=0.9)
        # # if listy[j] in kgword_dict:
        # KG_test[i].append(x[0])

# print(KG_test)
# print(X_train)
# for i in 
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
count = 0.0
for i in range(len(X_train)):
    count = count + len(X_train[i])

count = count / len(X_train)
print(count)


# print(X_train)
count1 = 0.0
for i in range(len(KG_train)):
    count1 = count1 + len(KG_train[i])

count1 = count1 / len(KG_train)
print(count1)

vocab_size = len(tokenizer.word_index) + 1

maxlenw = int(count)
maxlenk = int(count1)
KG_train = pad_sequences(KG_train, padding='post', maxlen=maxlenk)
KG_test = pad_sequences(KG_test, padding='post', maxlen=maxlenk)
X_train = pad_sequences(X_train, padding='post', maxlen=maxlenw)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlenw)

print(KG_train)
# print(type(X_train), X_train[4977])
embeddings_dictionary = dict()
# print(KG_train)
# print(len(X_train))
# print(len(X_test))
# print(len(y_train))
# print(len(y_test))

# text_embedding = np.zeros((len(tokenizer.word_index) + 1, 300))
# for word, i in tokenizer.word_index.items():
#    text_embedding[i] = nlp(word).vector
# model.add(Conv1D(filters, kernel_size=kernel_size,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
# model.add(Conv1D(filters, kernel_size=kernel_size,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
# model.add(Conv1D(filters, kernel_size=kernel_size,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
embedding_matrix = np.zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
leng = len(entity) + len(relation)
kg_embedding_matrix = np.empty((leng, 100))
j = 0
for i in range(len(entity)):
    en = np.array(entity[i])
    kg_embedding_matrix[j] = en
    j = j + 1
for i in range(len(relation)):
    en = np.array(relation[i])
    kg_embedding_matrix[j] = en
    j = j + 1
kg_vocab_size = kg_embedding_matrix.shape[0]
# print(kg_vocab_size)
# print(enlab)
a = []
# for i in enlab:
#     a.append(str(i[0]))
# x = difflib.get_close_matches('oscurr',a, n=5, cutoff=0.7)
# print(x)

# result = 1 - spatial.distance.cosine(embeddings_dictionary.get('oscar'), entity[208].tolist())

# Word Enbedding Model 
input_word = Input(shape=(maxlenw,))
embedd = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(input_word)
LTSM_word = LSTM(64)(embedd)
model_word = Model(inputs=input_word, outputs=LTSM_word)
# dense_1 = Dense(max)

conv = Conv1D(filters=300, )

# input_KG = Input(shape=(maxlenk,))
# embedd_KG = Embedding(kg_vocab_size, 100, weights=[kg_embedding_matrix], trainable=False)(input_KG)
# LTSM_kg = LSTM(64)(embedd_KG)
# model_kg = Model(inputs=input_KG, outputs=LTSM_kg)

# combined_outputs = concatenate([model_word.output, model_kg.output])
dense_concat = Dense(max_amojis, activation='sigmoid')(model_word.output)
main_model = Model(inputs=[model_word.input], outputs=dense_concat)
main_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='val_acc', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False) 
his = main_model.fit(X_train, y_train, batch_size=64, epochs=70, validation_split=0.2, callbacks=[checkpoint1])
# print(result[1])
main_model.summary()
result = main_model.evaluate(X_test, y_test, verbose = 1)
print(result[1])


# deep_inputs = Input(shape=(maxlen,))
# embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=False)(deep_inputs)
# # kg_embedding_layer = Embedding(kg_vocab_size, 100, weights=[kg_embedding_matrix], trainable=False)(deep_inputs)
# # concat_layer = Concatenate()([embedding_layer, kg_embedding_layer])
# #LSTM_Layer_1 = LSTM(32)(concat_layer)

# LSTM_Layer_1 = LSTM(64)(embedding_layer)
# # LSTM_Layer_2 = Layer(LSTM(32)(kg_embedding_layer))
# # LSTM_layer = Bidirectional(LSTM_Layer_1)
# dense_layer_1 = Dense(max_amojis, activation='softmax')(LSTM_Layer_1)
# model = Model(inputs=deep_inputs, outputs=dense_layer_1)

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# # from tensorflow.keras.utils import plot_model
# # plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)
# checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='val_acc', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False) 
# history = model.fit(X_train, y_train, batch_size=64, epochs=20, validation_split=0.2, callbacks=[checkpoint1])

# # del model
main_model = load_model('best_model1.hdf5')
X_train = X_train[len(X_train)-5:len(X_train)-1]
KG_train = KG_train[len(KG_train)-5:len(KG_train)-1]
x = main_model.predict([X_train])

for i in range(len(emojis)):
    emojis[i] = [emojis[i][0], i]

print(x)
# emo ['', index]
for arr in x:
    # []

    top_2_idx = np.argsort(arr)[-5:]
    top_2_values = [arr[i] for i in top_2_idx]
    print(top_2_idx)
    emoji_pred = []
    for vec in emojis:
        for i in top_2_idx:
            if (vec[1] == i):
                emoji_pred.append(vec[0])
    # emojis = map(lambda x: EMOJIS[x], top_2_idx)
    print(emoji.emojize(' '.join(emoji_pred), use_aliases=True))

glove_file.close()