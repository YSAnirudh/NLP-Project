# -*- coding: utf-8 -*-

""" Use torchMoji to predict emojis from a single text input
"""

from __future__ import print_function, division, unicode_literals
import json
import argparse
import pandas as pd
import numpy as np
import emoji
from collections import Counter
from time import time
import concurrent.futures
from tqdm import tqdm
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

# Emoji map in emoji_overview.png
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


dataset = pd.read_csv('replies3.csv', encoding="utf-8").values.tolist()
num_threads = 10
parent_tweet = dataset[0][1]
#print(parent_tweet)
reply_tweets = []
reply_tweets = dataset[1:]

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

def get_emoji(index):
    if (index >= len(reply_tweets)):
        return
    x1 = time()
    #print(index)
    text = reply_tweets[index][1]
    maxlen = 200

    # Tokenizing using dictionary
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, maxlen)

    # Loading model
    model = torchmoji_emojis(PRETRAINED_PATH)
    # Running predictions
    tokenized, _, _ = st.tokenize_sentences([text])
    # Get sentence probability
    prob = model(tokenized)[0]

    # Top emoji id
    emoji_ids = top_elements(prob, 5)
    #print("Tweet " + str(index) + " Time:", str(time() - x1))
    return emoji_ids

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--text', type=str, required=True, help="Input text to emojize")
    # argparser.add_argument('--maxlen', type=int, default=30,help="Max length of input text")
    # args = argparser.parse_args()
    emoji_id_list = []  # this will store all the emoji index which can in the dataset
    t_sum = 0
    # count = 0
    # for tweets in dataset:
    #     if count != 0:
    #         temp_tweets = tweets
    #         reply_tweets.append(temp_tweets)
    #         print(reply_tweets)
    #     count = count + 1

    ### PARALLEL

    startTime = time()
    processes = []
    intervals = []
    for i in range(len(reply_tweets)):
        # if (i % num_threads == 0):
        intervals.append(i)
    print(intervals)
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for (i, emoji_ids) in tqdm(zip(intervals, executor.map(get_emoji, intervals))):
            pass
    endTime = time()
    print("Multi time:", endTime - startTime)


     ### SERIAL
    i = 0
    x = time()
    # for index in tqdm(range(len(reply_tweets))):
    #     x1 = time()
    #     text = reply_tweets[index][1]
    #     maxlen = 200

    #     # Tokenizing using dictionary
    #     with open(VOCAB_PATH, 'r') as f:
    #         vocabulary = json.load(f)

    #     st = SentenceTokenizer(vocabulary, maxlen)

    #     # Loading model
    #     model = torchmoji_emojis(PRETRAINED_PATH)
    #     # Running predictions
    #     tokenized, _, _ = st.tokenize_sentences([text])
    #     # Get sentence probability
    #     prob = model(tokenized)[0]

    #     # Top emoji id
    #     emoji_ids = top_elements(prob, 5)
    #     emoji_id_list.extend(emoji_ids)
    #     #print("Tweet " + str(i) + " Time:", str(time() - x1))
    #     i = i + 1
    #     # for i in range(5):
    #     #     # only taking the most prob top 5 emojis for each sentence
    #     #     emoji_id_list.append(emoji_ids[i])
    y = time()
    # this will return the count of who many times a particular emoji(index) occured
    w = Counter(emoji_id_list)
    print(y - x)
    final_emoji_ids = [index for index, key in w.most_common(5)]
    # print(final_emoji_ids) # this wil print the index of top 5 most occured emojis
    # map to emojis
    emojis = map(lambda x: EMOJIS[x], final_emoji_ids)
    print(emoji.emojize(' '.join(emojis), use_aliases=True))
    # print(emoji.emojize("{} {}".format(text, ' '.join(emojis)), use_aliases=True))
