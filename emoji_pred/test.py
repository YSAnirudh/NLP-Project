from torchmoji_label import TweetLabel
import pandas as pd
if __name__ == '__main__':
    x = TweetLabel(parent_csv = 'parents.csv', replies_csv = 'replies.csv', parallel=True)
    x.parse_csv()
    x.predict()
    x.write_results_to_csv()
    # parents_df = pd.read_csv('replies.csv', lineterminator='\n')
    # parents_df.drop('Unnamed: 0', axis=1, inplace=True)
    # reps = pd.read_csv('repliesOscars.csv', lineterminator='\n')
    # reps.drop('Unnamed: 0', axis=1, inplace=True)

    # df = pd.DataFrame()
    # df = pd.concat([parents_df, reps])
    # df.reset_index(drop=True, inplace=True)
    # df.to_csv('main-replies.csv')