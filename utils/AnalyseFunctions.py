import re
import os
import nltk

from utils.Utils import readDBIntoTweetList
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

## Tokenizers
# break tweets into sentences
sentDetector = nltk.data.load('tokenizers/punkt/english.pickle')
# tweet tokenizer
tweetTokenizer = TweetTokenizer()

# Keywords && stopwords
keywords = ['hosting', "host", "hosts", 'won', 'winner', 'wins', 'presented', 'presenter', 'dressed', 'dress', 'best-dressed',
            'suit']
customizedStopwords = ['golden', 'globe', 'globes', 'goldenglobes', 'goldenglobe']
stopwordlist = set(stopwords.words('english'))
for cstopword in customizedStopwords:
    stopwordlist.add(cstopword)
## RE
keywordsCleanerRE = re.compile("|".join(keywords), re.IGNORECASE)
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

def findHost():
    hostWords = ["host", "hosts", "hosting"]
    res = {}
    for tweet in cleanedTweetList:
        for hw in hostWords:
            if hw in tweet:
                tmp = tweet.lower()
                tokens = tweetTokenizer.tokenize(tmp)
                usefulTokens = [w for w in tokens if not w in stopwordlist]
                for k in nltk.bigrams(usefulTokens):
                    if k in res:
                        res[k] += 1
                    else:
                        res[k] = 1
    sortedDict = sorted(res.items(), key=lambda entry: entry[1], reverse=True)

    return sortedDict[0], sortedDict[1]

print(findHost())

def findwinner(i):
    file = open(datapath +"/AwardCategories2013.txt")
    lines = file.read().split("\n")
    line = lines[i]
    string = line.split("-")
    if len(string) == 1:
        string.append(" ")
    awardString = string[0]
    catagoryString = string[1]
    awardstring = awardString.lower()
    catagorystring = catagoryString.lower()
    awardWords = tweetTokenizer.tokenize(awardstring)
    catagorywords = tweetTokenizer.tokenize(catagorystring)
    if len(catagorywords) == 0:
        catagorywords.append(" ")
    words = awardWords + catagorywords
    res = {}
    for tweet in cleanedTweetList:


        for aw in nltk.bigrams(awardWords):
            if aw[0] not in stopwordlist and aw[1] not in stopwordlist:
                for cw in catagorywords:
                    tweet = tweet.lower()
                    if "actor" in awardWords and "actress" in tweet or "actress" in awardWords and "actor" in tweet:
                        continue

                    if aw[0] in tweet and aw[1] in tweet and (cw in tweet or cw == ' '):
                        tmp = tweet.lower()
                        tokens = tweetTokenizer.tokenize(tmp)
                        usefulTokens = [w for w in tokens if not w in stopwordlist]
                        for k in nltk.bigrams(usefulTokens):
                            if k[0] not in words and k[1] not in words:
                                if k in res:
                                    res[k] += 1
                                else:
                                    res[k] = 1
    sortedDict = sorted(res.items(), key=lambda entry: entry[1], reverse=True)

    return sortedDict

print(findwinner(12))


# if __name__ == '__main__':
#     tweetList = readDBIntoTweetList("gg2013")
#     cleanedTweetList = tweetsCleaner(tweetList)
#     print("test")

    # tknzr = TweetTokenizer()
    # for tweet in tweetList:
    #     fuck = sentDetector.tokenize(tweet.get_text())
    #     tmp = tknzr.tokenize(tweet.get_text())
    #     print("fuck")
