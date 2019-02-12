import re
import os
import nltk
import numpy as np
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from utils.Utils import readDBIntoTweetList
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

## Tokenizers
# break tweets into sentences
sentDetector = nltk.data.load('tokenizers/punkt/english.pickle')
# tweet tokenizer
tweetTokenizer = TweetTokenizer()

# Keywords && stopwords
keywords = ['hosting', "host", "hosts", 'won', 'best', 'winner', 'wins', 'presented', 'presenter', 'dressed', 'dress', 'best-dressed',
            'suit', 'win', 'limited']
nominee_keywords = ['nominee', 'nominees', 'who', 'which']
customizedStopwords = ['golden', 'globe', 'globes', 'goldenglobes', 'goldenglobe']
stopwordlist = set(stopwords.words('english'))
for cstopword in customizedStopwords:
    stopwordlist.add(cstopword)
## RE
keywordsCleanerRE = re.compile("|".join(stopwordlist), re.IGNORECASE)
retweetCleanerRE = re.compile('RT', re.IGNORECASE)

datapath = os.path.abspath(os.path.dirname(os.getcwd())) + '/data'




def tweetsCleaner(tweetList):
    cleanedTweetList = []
    cnt = 0
    for tweet in tweetList:
        #cnt += 1
        #print(cnt)
        sentences = sentDetector.tokenize(tweet.get_text())
        if not retweetCleanerRE.search(sentences[0]):
            for s in sentences:
                if keywordsCleanerRE.search(s):
                    cleanedTweet = re.sub("[^a-zA-Z0-9 ]", "", tweet.get_text())
                    cleanedTweetList.append(cleanedTweet)

    return cleanedTweetList

cleanedTweetList = tweetsCleaner(readDBIntoTweetList("gg2013"))


print(len(cleanedTweetList))

def findawardsname():
    res = {}
    temp_len = 0
    for tweet in cleanedTweetList:
        temp_len = 0
        tweet_l = tweet.lower()
        if 'best' in tweet_l:
            if 'comedy' in tweet_l:
                award_index0 = tweet_l.find('best')
                award_index1 = tweet_l.find('comedy')
                award = tweet_l[award_index0:award_index1 + 6]
                tokens = tweetTokenizer.tokenize(award)
                if len(tokens) >= 4:

                    if len(tokens) > temp_len:
                        temp_len = max(temp_len, len(tokens))
                        a_i0 = award_index0
                        a_i1 = award_index1 + 6


        if 'best' in tweet_l:
            if 'drama' in tweet_l:
                award_index0 = tweet_l.find('best')
                award_index1 = tweet_l.find('drama')
                award = tweet_l[award_index0:award_index1 + 5]
                tokens = tweetTokenizer.tokenize(award)
                if len(tokens) >= 4:

                    if len(tokens) > temp_len:
                        temp_len = max(temp_len, len(tokens))
                        a_i0 = award_index0
                        a_i1 = award_index1 + 5

        if 'best' in tweet_l:
            if 'picture' in tweet_l:
                award_index0 = tweet_l.find('best')
                award_index1 = tweet_l.find('picture')
                award = tweet_l[award_index0:award_index1 + 7]
                tokens = tweetTokenizer.tokenize(award)
                if len(tokens) >= 4:

                    if len(tokens) > temp_len:
                        temp_len = max(temp_len, len(tokens))
                        a_i0 = award_index0
                        a_i1 = award_index1 + 7

        if 'best' in tweet_l:
            if 'television' in tweet_l:
                award_index0 = tweet_l.find('best')
                award_index1 = tweet_l.find('television')
                award = tweet_l[award_index0:award_index1 + 10]
                tokens = tweetTokenizer.tokenize(award)
                if len(tokens) >= 4:

                    if len(tokens) > temp_len:
                        temp_len = max(temp_len, len(tokens))
                        a_i0 = award_index0
                        a_i1 = award_index1 + 10

        if temp_len >= 4:
            award = tweet_l[a_i0: a_i1]
            if award in res:
                res[award] += 1
            else:
                res[award] = 1


    sortedDict = sorted(res.items(), key=lambda entry: entry[1], reverse=True)
    return sortedDict

print(findawardsname())
