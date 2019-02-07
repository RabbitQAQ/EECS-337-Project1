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
    awardWords = tweetTokenizer.tokenize(awardstring)
    catagorywords = tweetTokenizer.tokenize(catagorystring)
    wordtovec = []
    if len(catagorywords) == 0:
        catagorywords.append(" ")
    words = awardWords + catagorywords

    if len(awardWords) >= 8:
        res = {}
        for tweet in cleanedTweetList:
            tweet = tweet.lower()
            w = tweetTokenizer.tokenize(tweet)
            wordtovec.append(w)


            # if "foreign language" in tweet:
            #     print(tweet)

            for aw in nltk.trigrams(awardWords):
                if aw[0] not in stopwordlist and aw[1] not in stopwordlist and aw[2] not in stopwordlist:
                    for cw in catagorywords:

                        if "actor" in awardWords and ("actress" in tweet or "actor" not in tweet):
                            continue
                        if "actress" in awardWords and ("actor" in tweet or "actress" not in tweet):
                            continue

                        if aw[0] in tweet and aw[1] in tweet and aw[2] in tweet and (cw in tweet or cw == ' '):
                            tmp = tweet.lower()
                            tokens = tweetTokenizer.tokenize(tmp)
                            usefulTokens = [w for w in tokens if not w in stopwordlist]

                            try:
                                aw0_pos = word_tfidf.index(aw[0])
                            except:
                                aw0_pos = 0
                            try:
                                aw1_pos = word_tfidf.index(aw[1])
                            except:
                                aw1_pos = 0
                            try:
                                aw2_pos = word_tfidf.index(aw[2])
                            except:
                                aw2_pos = 0
                            for k in nltk.bigrams(usefulTokens):
                                if k[0] not in awardWords and k[1] not in awardWords:
                                    if k in res:
                                        res[k] += 1 * weight[i][aw0_pos] * weight[i][aw1_pos] * weight[i][aw2_pos]
                                    else:
                                        res[k] = 1 * weight[i][aw0_pos] * weight[i][aw1_pos] * weight[i][aw2_pos]
        sortedDict = sorted(res.items(), key=lambda entry: entry[1], reverse=True)



    if  len(awardWords) < 8  or len(sortedDict) == 0:
        res = {}
        for tweet in cleanedTweetList:
            tweet = tweet.lower()
            w = tweetTokenizer.tokenize(tweet)
            wordtovec.append(w)

            # if "game change" in tweet:
            #     print(tweet)

            for aw in nltk.bigrams(awardWords):
                if aw[0] not in stopwordlist and aw[1] not in stopwordlist:
                    for cw in catagorywords:

                        if "actor" in awardWords and ("actress" in tweet or "actor" not in tweet):
                            continue
                        if "actress" in awardWords and ("actor" in tweet or "actress" not in tweet):
                            continue

                        if aw[0] in tweet and aw[1] in tweet and (cw in tweet or cw == ' '):
                            tmp = tweet.lower()
                            tokens = tweetTokenizer.tokenize(tmp)
                            usefulTokens = [w for w in tokens if not w in stopwordlist]

                            try:
                                aw0_pos = word_tfidf.index(aw[0])
                            except:
                                aw0_pos = 0
                            try:
                                aw1_pos = word_tfidf.index(aw[1])
                            except:
                                aw1_pos = 0
                            for k in nltk.bigrams(usefulTokens):
                                if k[0] not in awardWords and k[1] not in awardWords:
                                    if k in res:
                                        res[k] += 1 * weight[i][aw0_pos] * weight[i][aw1_pos]
                                    else:
                                        res[k] = 1 * weight[i][aw0_pos] * weight[i][aw1_pos]
        sortedDict = sorted(res.items(), key=lambda entry: entry[1], reverse=True)

    if len(sortedDict) == 0:
        res = {}
        for tweet in cleanedTweetList:
            tweet = tweet.lower()
            w = tweetTokenizer.tokenize(tweet)
            wordtovec.append(w)

            # if "game change" in tweet:
            #     print(tweet)

            for aw in awardWords:
                if aw not in stopwordlist:
                    for cw in catagorywords:

                        if "actor" in awardWords and ("actress" in tweet or "actor" not in tweet):
                            continue
                        if "actress" in awardWords and ("actor" in tweet or "actress" not in tweet):
                            continue

                        if aw in tweet and (cw in tweet or cw == ' '):
                            tmp = tweet.lower()
                            tokens = tweetTokenizer.tokenize(tmp)
                            usefulTokens = [w for w in tokens if not w in stopwordlist]

                            try:
                                aw0_pos = word_tfidf.index(aw)
                            except:
                                aw0_pos = 0

                            for k in nltk.bigrams(usefulTokens):
                                if k[0] not in awardWords and k[1] not in awardWords:
                                    if k in res:
                                        res[k] += 1 * weight[i][aw0_pos]
                                    else:
                                        res[k] = 1 * weight[i][aw0_pos]
        sortedDict = sorted(res.items(), key=lambda entry: entry[1], reverse=True)

    # model = word2vec.Word2Vec(wordtovec, workers = 2, size = 200, min_count = 10, window = 5, sample = 0.001)


    #
    # for tem in sortedDict:
    #     word_1 = tem[0][0]
    #     word_2 = tem[0][1]

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
