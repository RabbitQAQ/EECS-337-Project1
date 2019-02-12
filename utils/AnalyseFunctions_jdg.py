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


print(len(cleanedTweetList))

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
    awardstring = awardstring.replace('limited', 'mini')
    catagorystring = catagoryString.lower()
    Ismovie = 1
    if 'actor' in awardstring or 'actress' in awardstring or 'director' in awardstring or 'srceenplay' in awardstring or 'cecil' in awardstring or 'score' in awardstring:
        Ismovie = 0
    # awardWords = tweetTokenizer.tokenize(awardstring)
    # categorywords = tweetTokenizer.tokenize(catagorystring)
    awardWords = [w for w in tweetTokenizer.tokenize(awardstring) if not w in stopwordlist]
    categorywords = [w for w in tweetTokenizer.tokenize(catagorystring) if not w in stopwordlist]
    if len(categorywords) == 0:
        categorywords.append(" ")
    words = awardWords + categorywords

    sortedDict = {}

    if len(awardWords) >= 8:
        sortedDict = findWinnerInNgrams(i, awardWords, categorywords, 3)

    if len(awardWords) < 8 or len(sortedDict) == 0:
        sortedDict = findWinnerInNgrams(i, awardWords, categorywords, 2)

    if len(sortedDict) == 0:
        sortedDict = findWinnerInNgrams(i, awardWords, categorywords, 1)


    winner = sortedDict[0][0][0] + ' ' + sortedDict[0][0][1]

    diff = (sortedDict[0][1] - sortedDict[1][1]) / sortedDict[0][1]
    if Ismovie == 0 and diff < 0.10:
        file1 = open(datapath + "/name2013.txt")
        names = file1.read().split("\n")
        n0 = 0
        n1 = 0
        for name in names:
            name = name.lower()
            if sortedDict[0][0][0] in name and sortedDict[0][0][1] in name:
                n0 += 1
            if sortedDict[1][0][0] in name and sortedDict[1][0][1] in name:
                n1 += 1

        if n1 > n0:
            winner = sortedDict[1][0][0] + ' ' + sortedDict[1][0][1]

    if Ismovie == 1:
        if diff < 0.10 and sortedDict[0][0][0] == sortedDict[1][0][1]:
            winner = sortedDict[1][0][0] + ' ' + sortedDict[1][0][1] + ' ' + sortedDict[0][0][1]
        if diff < 0.10 and sortedDict[1][0][0] == sortedDict[0][0][1]:
            winner = sortedDict[0][0][0] + ' ' + sortedDict[0][0][1] + ' ' + sortedDict[1][0][1]
        if sortedDict[0][0][0] == 'wins' or sortedDict[0][0][0] == 'goes' or sortedDict[0][0][0] == 'movie' or sortedDict[0][0][0] == 'flim':
            winner = sortedDict[0][0][1]
        if sortedDict[0][0][1] == 'wins' or sortedDict[0][0][1] == 'goes' or sortedDict[0][0][1] == 'movie' or sortedDict[0][0][1] == 'flim':
            winner = sortedDict[0][0][0]


    return winner

# def ie_preprocess(document):
#     document = ' '.join([i for i in document.split() if i not in stopwordlist])
#     sentences = nltk.sent_tokenize(document)
#     sentences = [nltk.word_tokenize(sent) for sent in sentences]
#     sentences = [nltk.pos_tag(sent) for sent in sentences]
#     return sentences

# def extract_names(document):
#
#     sentences = ie_preprocess(document)
#     for tagged_sentence in sentences:
#         for chunk in nltk.ne_chunk(tagged_sentence):
#             if type(chunk) == nltk.tree.Tree:
#                 if chunk.label() == 'PERSON':
#                     na = ' '.join([c[0] for c in chunk])
#                     if len(chunk) == 2:
#                         if na in name:
#                             name[na] += 1
#                         else:
#                             name[na] = 1

# def get_name():
#     name = {}
#     for tweet in cleanedTweetList:
#         extract_names(tweet)
#     return name


def findWinnerInNgrams(i, awardWords, categoryWords, n):
    res = {}

    for tweet in cleanedTweetList:

        tweet_l = tweet.lower()
        # w = tweetTokenizer.tokenize(tweet)
        # wordtovec.append(w)

        for aw in nltk.ngrams(awardWords, n):
            flag = True
            for tw in aw:
                if tw in stopwordlist or not tw in tweet_l:
                    flag = False
            if flag:
                for cw in categoryWords:

                    if "actor" in awardWords and ("actress" in tweet or "actor" not in tweet):
                        continue
                    if "actress" in awardWords and ("actor" in tweet or "actress" not in tweet):
                        continue

                    if flag and (cw in tweet or cw == ' '):

                        tokens = tweetTokenizer.tokenize(tweet_l)
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
    # sortedDictname = sorted(name.items(), key=lambda entry: entry[1], reverse=True)
    # model = word2vec.Word2Vec.load("model1")
    # y1 = model.most_similar(sortedDict[0][0][0], topn=30)
    # y2 = model.most_similar(sortedDict[0][0][1], topn=30)
    #
    # for i in y1:
    #     for key, value in name.items():
    #         if i[0] in key.lower() and value > 10:
    #             keyl = key.lower()
    #             keyw = keyl.split(" ")
    #             temp = model.most_similar(keyw[0], topn=10)
    #             if model.similarity(keyw[0], keyw[1]) >= temp[1][1]:
    #                 print(key)
    # for i in y2:
    #     for key, value in name.items():
    #         if i[0] in key.lower() and value > 10:
    #             keyl = key.lower()
    #             keyw = keyl.split(" ")
    #             temp = model.most_similar(keyw[0], topn=10)
    #             if model.similarity(keyw[0], keyw[1]) >= temp[1][1]:
    #                 print(key)
    return sortedDict


for i in range (0, 26):
    print(findwinner(i))

# name = {}
# name_final = []
# wordtovec = []
# model = word2vec.Word2Vec.load("model2")
# for tweet in cleanedTweetList:
#     extract_names(tweet, name)
#
# for key, value in name.items():
#     keyl = key.lower()
#     keyw = keyl.split(" ")
#     if keyw[0] in model and keyw[1] in model:
#         temp0 = model.most_similar(keyw[0], topn=100)
#         temp1 = model.most_similar(keyw[1], topn=100)
#         if value > 5:
#             if model.similarity(keyw[0], keyw[1]) >= temp0[2][1] or model.similarity(keyw[1], keyw[0]) >= temp1[2][1]:
#                 name_final.append(key)
#
# file1 = open(datapath +"/name2013.txt", 'w')
# for i in name_final:
#     file1.write(i)
#     file1.write('\n')
# file1.close()

# for tweet in cleanedTweetList:
#     tweet = tweet.lower()
#     w = tweetTokenizer.tokenize(tweet)
#     wordtovec.append(w)
#
# model = word2vec.Word2Vec(wordtovec, workers = 2, size = 200, min_count = 5, window = 5, sample = 0.001)
# model.save("model2")
# #
model = word2vec.Word2Vec.load("model2")
# y1 = model.most_similar("kathryn", topn=10)
# y2 = model.most_similar("david", topn=10)
# y3 = model.most_similar("tommy", topn=10)
# y4 = model.most_similar("danny", topn=10)
# print(y3,y4,y1,y2)
# # 20个最相关的
# for i in y1:
#     for key, value in name.items():
#         if i[0] in key.lower() and value > 10:
#             keyl = key.lower()
#             keyw = keyl.split(" ")
#             temp = model.most_similar(keyw[0], topn=10)
#             if model.similarity(keyw[0], keyw[1]) >= temp[1][1]:
#                 print(key)
# for i in y2:
#     for key, value in name.items():
#         if i[0] in key.lower() and value > 10:
#             keyl = key.lower()
#             keyw = keyl.split(" ")
#             temp = model.most_similar(keyw[0], topn=10)
#             if model.similarity(keyw[0], keyw[1]) >= temp[1][1]:
#                 print(key)



pass
# if __name__ == '__main__':
#     tweetList = readDBIntoTweetList("gg2013")
#     cleanedTweetList = tweetsCleaner(tweetList)
#     print("test")

    # tknzr = TweetTokenizer()
    # for tweet in tweetList:
    #     fuck = sentDetector.tokenize(tweet.get_text())
    #     tmp = tknzr.tokenize(tweet.get_text())
    #     print("fuck")
