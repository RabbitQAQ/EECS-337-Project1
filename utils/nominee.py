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
customizedStopwords = ['golden', 'globe', 'globes', 'goldenglobes', 'goldenglobe', '-']
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


def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stopwordlist])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def extract_names(document):
    count = 0
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            qwe = type(chunk)
            wer = nltk.tree.Tree

            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    na = ' '.join([c[0] for c in chunk])

                    if len(chunk) == 2:
                        count += 1
                        if na in nominee_name:
                            nominee_name[na] += 1
                        else:
                            nominee_name[na] = 1

    if count >= 5:
        nominee_tweet.append(document)


nominee_tweet = []
nominee_name = {}

for tweet in cleanedTweetList:
    tweet = tweet.lower()
    if "claire danes" in tweet:
        extract_names(tweet)

# for tweet in nominee_tweet:
#     tweet = tweet.lower()
#
#     if "claire danes" in tweet:
#         extract_names(tweet)
print(nominee_name)