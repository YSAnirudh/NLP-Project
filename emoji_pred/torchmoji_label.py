import json
import argparse
import pandas as pd
import numpy as np
import emoji
import copy
from collections import Counter
from time import time
import concurrent.futures
from tqdm import tqdm
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

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

class TweetLabel():
    def __init__(self, parent_csv, replies_csv, parallel:bool = False):
        self.parent_csv = parent_csv
        self.replies_csv = replies_csv
        self.parallel = parallel
        self.reply_tweets_w_parent_id = []
        self.result = []

    def top_elements(self, array, k):
        ind = np.argpartition(array, -k)[-k:]
        return ind[np.argsort(array[ind])][::-1]

    def write_results_to_csv(self):
        tweets_df2 = pd.DataFrame(self.result, columns=['Tweet_ID', 'Tweet_Text', 'Emojis'])
        tweets_df2.drop_duplicates(inplace=True)
        tweets_df2.to_csv('labelled_data.csv',encoding="utf-8")

    def get_emoji(self, rep_text):
        x1 = time()
        #print(index)
        text = rep_text[1]
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
        emoji_ids = self.top_elements(prob, 5)
        #print("Tweet " + str(index) + " Time:", str(time() - x1))
        return emoji_ids

    def parse_csv(self):
        parents_df = pd.read_csv(self.parent_csv, lineterminator='\n')
        parents_df.drop('Unnamed: 0', axis=1, inplace=True)
        replies_df = pd.read_csv(self.replies_csv, lineterminator='\n')
        replies_df.drop('Unnamed: 0', axis=1, inplace=True)
        self.reply_tweets_w_parent_id = [None for i in range(len(list(parents_df['Tweet Id'])))]
        i = 0
        for id in parents_df['Tweet Id']:
            # replies_df.loc[replies_df['Parent_Id'] == id]
            self.reply_tweets_w_parent_id[i] = [id, list(replies_df.loc[replies_df['Parent_Id'] == id]['Text']), str(list(parents_df.loc[parents_df['Tweet Id'] == id]['Text'])[0])]
            i = i + 1

        # print(replies_df)
    
    def predict(self):
        # argparser = argparse.ArgumentParser()
        # argparser.add_argument('--text', type=str, required=True, help="Input text to emojize")
        # argparser.add_argument('--maxlen', type=int, default=30,help="Max length of input text")
        # args = argparser.parse_args()
          # this will store all the emoji index which can in the dataset
        #self.reply_tweets_w_parent_id[tweed_id] = [parentID, [replies]]

        for i in tqdm(range(len(self.reply_tweets_w_parent_id))):
            if (i < 2000 or i > 4000):
                break
            emoji_id_list = []
            # print("Extracting for:", self.reply_tweets_w_parent_id[i][0])
            reply_tweets = self.reply_tweets_w_parent_id[i][1]
            if (self.parallel):
                intervals = []
                for iD in range(len(reply_tweets)):
                    intervals.append(reply_tweets[iD])
                with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                    for (indexy, emoji_ids) in zip(intervals, executor.map(self.get_emoji, intervals)):
                        emoji_id_list.extend(emoji_ids)
            else:
                # x = time()
                for index in range(len(reply_tweets)):
                    # x1 = time()
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
                    emoji_ids = self.top_elements(prob, 5)
                    emoji_id_list.extend(emoji_ids)
                    #print("Tweet " + str(i) + " Time:", str(time() - x1))
                # print("Serial Time:",time()-x)
            
            w = Counter(emoji_id_list)
            final_emoji_ids = [index for index, key in w.most_common(5)]
            # print(final_emoji_ids) # this wil print the index of top 5 most occured emojis
            # map to emojis
            emojis = list(map(lambda x: EMOJIS[x], final_emoji_ids))
            result_list = []
            result_list.append(self.reply_tweets_w_parent_id[i][0])
            result_list.append(self.reply_tweets_w_parent_id[i][2])
            emoji_str = ' '.join(emojis)
            result_list.append(emoji_str)
            self.result.append(result_list)
            # print(emoji.emojize(' '.join(emojis), use_aliases=True))

            