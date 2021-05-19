from numpy import nan
from torchmoji_label import TweetLabel
import pandas as pd
if __name__ == '__main__':
    x = TweetLabel(parent_csv = 'main-parents.csv', replies_csv = 'main-replies.csv', parallel = True, clean_result_csv = True)
    x.parse_csv()
    x.predict()
    x.write_results_to_csv()
    
    #EMOJIS = [EMOJI for EMOJI in EMOJIS if EMOJI != '']
    # print(EMOJIS)
    # for i in EMOJIS:
    #     if (i != ''):
    #         print(i)


    # count = 0
    # for i in parents_df['Emojis']:
    #     if (len(str(i)) > 10):
    #         count = count + 1
    # print(count)
    # parents_df = pd.read_csv('main-parents.csv')
    # reps = pd.read_csv('parentsBenz.csv', lineterminator='\n')
    # reps.drop('Unnamed: 0', axis=1, inplace=True)

    # df = pd.DataFrame()
    # df = pd.concat([parents_df, reps])
    # df.drop('Unnamed: 0', axis=1, inplace=True)
    # df.reset_index(drop=True, inplace=True)

    # df.to_csv('main-parents.csv')