import re

import nltk

from utils.Utils import readDBIntoTweetList
from nltk.tokenize import TweetTokenizer

## Tokenizers
# break tweets into sentences
sentDetector = nltk.data.load('tokenizers/punkt/english.pickle')
# tweet tokenizer
tweetTokenizer = TweetTokenizer()

## RE
keywords = ['hosting', "host", "hosts", 'won', 'winner', 'wins', 'presented', 'presenter', 'dressed', 'dress', 'best-dressed',
            'suit']
keywordsCleanerRE = re.compile("|".join(keywords), re.IGNORECASE)
retweetCleanerRE = re.compile('RT', re.IGNORECASE)

def tweetsCleaner(tweetList):
    cleanedTweetList = []
    for tweet in tweetList:
        sentences = sentDetector.tokenize(tweet.get_text())
        if not retweetCleanerRE.search(sentences[0]):
            for s in sentences:
                if keywordsCleanerRE.search(s):
                    cleanedTweet = re.sub("[^a-zA-Z0-9 ]", "", tweet.get_text())
                    cleanedTweetList.append(cleanedTweet)

    return cleanedTweetList

def findHost():
    pass

if __name__ == '__main__':
    tweetList = readDBIntoTweetList("gg2013")
    cleanedTweetList = tweetsCleaner(tweetList)
    print("test")

    # tknzr = TweetTokenizer()
    # for tweet in tweetList:
    #     fuck = sentDetector.tokenize(tweet.get_text())
    #     tmp = tknzr.tokenize(tweet.get_text())
    #     print("fuck")
