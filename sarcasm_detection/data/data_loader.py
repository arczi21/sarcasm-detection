import pandas as pd


class SarcasticDataLoader:
    def __init__(self, data_frame):
        self.df = data_frame
        self.length = len(self.df)
        self.idx = 0

    def __len__(self):
        return self.length

    def get_headlines(self, begin, end):
        return list(self.df.loc[begin:end-1, 'headline'])

    def get_all_headlines(self):
        return self.get_headlines(0, self.length)

    def get_is_sarcastic(self, begin, end):
        return list(self.df.loc[begin:end-1, 'is_sarcastic'])

    def get_all_is_sarcastic(self):
        return self.get_is_sarcastic(0, self.length)

    def get_data(self, begin, end):
        return self.get_headlines(begin, end), self.get_is_sarcastic(begin, end)

    def get_all_data(self):
        return self.get_all_headlines(), self.get_all_is_sarcastic()