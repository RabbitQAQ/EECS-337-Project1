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
customizedStopwords = ['golden', 'globe', 'globes', 'goldenglobes', 'goldenglobe', '-', ',']
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

#TF-IDF
file = open(datapath + "/AwardCategories2013.txt")
lines = file.read().split("\n")
corpus = lines
#将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
#计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
#获取词袋中所有文本关键词
word_tfidf = vectorizer.get_feature_names()
print (word_tfidf)



transformer = TfidfTransformer()

#将词频矩阵X统计成TF-IDF值
tfidf = transformer.fit_transform(X)
weight=tfidf.toarray()

for wi in range (0, len(weight[0])):
    count = 0
    for wj in range (0, len(weight)):
        if weight[wj][wi] != 0:
            count += 1
            temp = wj
    if count == 1:
        weight[temp][wi] *= 10








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

#print(findHost())

def findwinner(i):
    file = open(datapath +"/AwardCategories2013.txt")
    lines = file.read().split("\n")
    line = lines[i]
    string = line.split("-")
    if len(string) == 1:
        string.append(" ")
    awardString = line
    catagoryString = string[1]
    awardstring = awardString.lower()
    catagorystring = catagoryString.lower()
    awardWords = [w for w in tweetTokenizer.tokenize(awardstring) if not w in stopwordlist]
    categorywords = [w for w in tweetTokenizer.tokenize(catagorystring) if not w in stopwordlist]
    if len(categorywords) == 0:
        categorywords.append(" ")
    words = awardWords + categorywords

    sortedDict = {}

    if len(awardWords) >= 8:
        sortedDict = findWinnerInNgrams(i, awardWords, categorywords, 3)

    if len(awardWords) < 8  or len(sortedDict) == 0:
        sortedDict = findWinnerInNgrams(i, awardWords, categorywords, 2)

    if len(sortedDict) == 0:
        sortedDict = findWinnerInNgrams(i, awardWords, categorywords, 1)

    return sortedDict

def findWinnerInNgrams(i, awardWords, categoryWords, n):
    res = {}
    for tweet in cleanedTweetList:
        tweet = tweet.lower()
        w = tweetTokenizer.tokenize(tweet)

        for aw in nltk.ngrams(awardWords, n):
            flag = True
            for tw in aw:
                if tw in stopwordlist or not tw in tweet:
                    flag = False
            if flag:
                for cw in categoryWords:

                    if "actor" in awardWords and ("actress" in tweet or "actor" not in tweet):
                        continue
                    if "actress" in awardWords and ("actor" in tweet or "actress" not in tweet):
                        continue

                    if flag and (cw in tweet or cw == ' '):
                        tmp = tweet.lower()
                        tokens = tweetTokenizer.tokenize(tmp)
                        usefulTokens = [w for w in tokens if not w in stopwordlist]
                        aw_pos = []
                        for tw in aw:
                            try:
                                aw_pos.append(word_tfidf.index(tw))
                            except:
                                aw_pos.append(0)
                        for k in nltk.bigrams(usefulTokens):
                            if k[0] not in awardWords and k[1] not in awardWords:
                                sum = 1
                                for tp in aw_pos:
                                    sum *= weight[i][tp]
                                if k in res:
                                    res[k] += sum
                                else:
                                    res[k] = sum
    sortedDict = sorted(res.items(), key=lambda entry: entry[1], reverse=True)

    return sortedDict

for i in range (0,26):
    print(findwinner(i))


# if __name__ == '__main__':
#     tweetList = readDBIntoTweetList("gg2013")
#     cleanedTweetList = tweetsCleaner(tweetList)
#     print("test")

    # tknzr = TweetTokenizer()
    # for tweet in tweetList:
    #     fuck = sentDetector.tokenize(tweet.get_text())
    #     tmp = tknzr.tokenize(tweet.get_text())
    #     print("fuck")
