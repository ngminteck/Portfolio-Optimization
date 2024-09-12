import nltk
from nltk.tokenize import word_tokenize
from urllib.parse import quote
import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os


class NewsSentiment:
    def __init__(self):
        # Define the download directory
        download_dir = os.path.join(os.path.dirname(__file__), '../nltk')

        # Download the 'punkt' package to the specified directory
        nltk.download('punkt', download_dir=download_dir)

        # Add the download directory to the NLTK data path
        nltk.data.path.append(download_dir)
        # Load the Loughran-McDonald word lists
        lm_dict = pd.read_csv('../nltk/Loughran-McDonald_MasterDictionary_1993-2023.csv')

        # Extract sentiment word lists
        self.negative_words = lm_dict[lm_dict['Negative'] > 0]['Word'].tolist()
        self.positive_words = lm_dict[lm_dict['Positive'] > 0]['Word'].tolist()
        self.uncertainty_words = lm_dict[lm_dict['Uncertainty'] > 0]['Word'].tolist()
        self.litigious_words = lm_dict[lm_dict['Litigious'] > 0]['Word'].tolist()
        self.strong_modal_words = lm_dict[lm_dict['Strong_Modal'] > 0]['Word'].tolist()
        self.weak_modal_words = lm_dict[lm_dict['Weak_Modal'] > 0]['Word'].tolist()
        self.constraining_words = lm_dict[lm_dict['Constraining'] > 0]['Word'].tolist()

    @staticmethod
    def fetch_news_titles_by_feedparser(search_queries):
        feed_list = []
        for query in search_queries:
            encoded_query = quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                published_date = datetime(*entry.published_parsed[:6])
                feed_list.append({"title": entry.title, "published_date": published_date})
        return feed_list

    def get_news_sentiment_score_by_feedparser(self, df, search_queries):

        sentiment_df = df.copy(deep=True)
        sentiment_df = sentiment_df.drop(columns=sentiment_df.columns)
        sentiment_df["SENTIMENT_NEGATIVE"] = 0.0
        sentiment_df["SENTIMENT_POSITIVE"] = 0.0
        sentiment_df["SENTIMENT_UNCERTAINTY"] = 0.0
        sentiment_df["SENTIMENT_LITIGIOUS"] = 0.0
        sentiment_df["SENTIMENT_STRONG_MODAL"] = 0.0
        sentiment_df["SENTIMENT_WEAK_MODAL"] = 0.0
        sentiment_df["SENTIMENT_CONSTRAINING"] = 0.0
        sentiment_df["START_DATE"] = sentiment_df.index.to_series().shift(1)

        all_titles = self.fetch_news_titles_by_feedparser(search_queries)

        for i in range(1, len(sentiment_df)):
            start_date = sentiment_df.iloc[i]["START_DATE"]
            end_date = sentiment_df.index[i]
            text_list = [entry["title"] for entry in all_titles if start_date <= entry["published_date"] <= end_date]
            # print(f"Row:{end_date} Title Size:{len(titles)}\n")

            for text in text_list:
                words = word_tokenize(text.upper())

                for word in words:
                    if word in self.negative_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_NEGATIVE'] += 1.0
                    if word in self.positive_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_POSITIVE'] += 1.0
                    if word in self.uncertainty_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_UNCERTAINTY'] += 1.0
                    if word in self.litigious_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_LITIGIOUS'] += 1.0
                    if word in self.strong_modal_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_STRONG_MODAL'] += 1.0
                    if word in self.weak_modal_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_WEAK_MODAL'] += 1.0
                    if word in self.constraining_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_CONSTRAINING'] += 1.0

        sentiment_df = sentiment_df.drop(['START_DATE'], axis=1, errors='ignore')

        df = pd.concat([df, sentiment_df], axis=1)

        return df

    @staticmethod
    def fetch_news_titles_by_beautifulsoup(search_queries, start_date, end_date):
        # Convert the Timestamp objects to strings
        start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

        # Convert the strings to datetime objects
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")

        # Extract the day, month, and year and convert them to strings
        start_day = str(start_date.day)
        start_month = str(start_date.month)
        start_year = str(start_date.year)

        end_day = str(end_date.day)
        end_month = str(end_date.month)
        end_year = str(end_date.year)

        feed_list = []
        for query in search_queries:
            encoded_query = quote(query)
            url = f"https://www.google.com.sg/search?q={encoded_query}&sca_esv=1b7c8f5127465da7&sca_upv=1&biw=2512&bih=1442&sxsrf=ADLYWIJH27Xp6Bx7pe9AhuQCqCP1ho-IdA%3A1725256624122&source=lnt&tbs=cdr%3A1%2Ccd_min%3A{start_month}%2F{start_day}%2F{start_year}%2Ccd_max%3A{end_month}%2F{end_day}%2F{end_year}&tbm=nws"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            for item in soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd'):
                feed_list.append(item.get_text())

        return feed_list

    def get_news_sentiment_by_beautifulsoup(self, df, search_queries):

        sentiment_df = df.copy(deep=True)
        sentiment_df = sentiment_df.drop(columns=sentiment_df.columns)
        sentiment_df["SENTIMENT_NEGATIVE"] = 0.0
        sentiment_df["SENTIMENT_POSITIVE"] = 0.0
        sentiment_df["SENTIMENT_UNCERTAINTY"] = 0.0
        sentiment_df["SENTIMENT_LITIGIOUS"] = 0.0
        sentiment_df["SENTIMENT_STRONG_MODAL"] = 0.0
        sentiment_df["SENTIMENT_WEAK_MODAL"] = 0.0
        sentiment_df["SENTIMENT_CONSTRAINING"] = 0.0
        sentiment_df["START_DATE"] = sentiment_df.index.to_series().shift(1)

        for i in range(1, len(sentiment_df)):
            start_date = sentiment_df.iloc[i]["START_DATE"]
            end_date = sentiment_df.index[i]
            text_list = self.fetch_news_titles_by_beautifulsoup(search_queries, start_date, end_date)

            for text in text_list:
                words = word_tokenize(text.upper())
                for word in words:
                    if word in self.negative_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_NEGATIVE'] += 1.0
                    if word in self.positive_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_POSITIVE'] += 1.0
                    if word in self.uncertainty_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_UNCERTAINTY'] += 1.0
                    if word in self.litigious_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_LITIGIOUS'] += 1.0
                    if word in self.strong_modal_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_STRONG_MODAL'] += 1.0
                    if word in self.weak_modal_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_WEAK_MODAL'] += 1.0
                    if word in self.constraining_words:
                        sentiment_df.loc[sentiment_df.index[i], 'SENTIMENT_CONSTRAINING'] += 1.0

        sentiment_df = sentiment_df.drop(['START_DATE'], axis=1, errors='ignore')

        df = pd.concat([df, sentiment_df], axis=1)

        return df
