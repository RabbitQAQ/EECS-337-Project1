import re
import os
import nltk
import spacy
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
                    cleanedTweet = re.sub("[^a-zA-Z0-9- ]", "", tweet.get_text())
                    cleanedTweetList.append(cleanedTweet)

    return cleanedTweetList

cleanedTweetList = tweetsCleaner(readDBIntoTweetList("gg2015"))


print(len(cleanedTweetList))

def findawardsname():
    res = {}
    temp_len = 0
    for tweet in cleanedTweetList:
        temp_len = 0
        tweet_l = tweet.lower()
        if 'tv' in tweet_l:
            tweet_l.replace('tv', 'television')
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

        if temp_len >= 5:
            award = tweet_l[a_i0: a_i1]
            word = award.split()
            if '-' in word or (word[-1] == 'television' or word[-1] == 'picture'):
                if award in res:
                    res[award] += 1
                else:
                    res[award] = 1

                flag = 0
                if 'actor' in word:
                    flag = 1
                if 'actress' in word:
                    flag = 2

                if flag == 1:
                    award.replace('actor', 'actress')
                    if award in res:
                        res[award] += 1
                    else:
                        res[award] = 1
                if flag == 2:
                    award.replace('actress', 'actor')
                    if award in res:
                        res[award] += 1
                    else:
                        res[award] = 1




    sortedDict = sorted(res.items(), key=lambda entry: entry[1], reverse=True)
    return sortedDict

# print(findawardsname())


def getjoke():
    spacyNLP = spacy.load('en')
    res = {}
    whosaid = {}
    bestdressedone = {}
    ans = []
    k = 10
    jokeword =['joke', 'lmao', 'lol', 'hhh', 'funny', '233']
    bestdressword = ['bestdress', 'best dress','best-dress']
    blacklist = ['best', 'dress', 'red', 'carpet', 'dressed']
    for tweet in cleanedTweetList:
        tweet_l = tweet.lower()

        for bw in bestdressword:
            if bw in tweet_l:
                entities = spacyNLP(tweet)
                for entity in entities.ents:
                    if entity.label_ == "PERSON":
                        if len(entity.text.split()) == 2:

                            if entity.text in bestdressedone:
                                bestdressedone[entity.text] += 1
                            else:
                                bestdressedone[entity.text] = 1
                # b_tokens = tweetTokenizer.tokenize(tweet_l)
                # b_usefulTokens = [w for w in b_tokens if not w in stopwordlist and not w in blacklist]
                # for bestdress in nltk.ngrams(b_usefulTokens, 2):
                #     if bestdress in bestdressedone:
                #         bestdressedone[bestdress] += 1
                #     else:
                #         bestdressedone[bestdress] = 1

        for jw in jokeword:
            if jw in tweet_l:
                tokens = tweetTokenizer.tokenize(tweet_l)
                usefulTokens = [w for w in tokens if not w in customizedStopwords]
                for who in nltk.ngrams(usefulTokens, 2):
                    if who in whosaid:
                        whosaid[who] += 1
                    else:
                        whosaid[who] = 1



                for joke in nltk.ngrams(usefulTokens, k):
                    if joke in res:
                        res[joke] += 1
                    else:
                        res[joke] = 1
    sortedDict = sorted(res.items(), key=lambda entry: entry[1], reverse=True)
    whosaid = sorted(whosaid.items(), key=lambda entry: entry[1], reverse=True)
    bestdressedone = sorted(bestdressedone.items(), key=lambda entry: entry[1], reverse=True)

    start = 1
    flag = 0
    temp_joke = []
    for i in range (0, len(sortedDict) - 1):
        if flag == 0:
            start = 1
            if len(temp_joke) != 0:
                joketonight = ''
                for word in temp_joke:
                    joketonight += word + ' '
                ans.append(joketonight)

            temp_joke = []
        if flag == 1:
            start == 0
        if start == 1:
            for tw in sortedDict[i][0]:
                temp_joke.append(tw)

            start = 0
            flag = 1

        for j in range (1, k):
            if sortedDict[i][0][j] != sortedDict[i + 1][0][j - 1] or sortedDict[i + 1][1] < 5:
                flag = 0


        if flag == 1:
             temp_joke.append(sortedDict[i + 1][0][k - 1])

    who = whosaid[0][0][0] + ' ' + whosaid[0][0][1]
    best_dressed = []
    for bd in bestdressedone[0:15]:
        temp0 = bd[0].split()

        best_dressed.append(temp0[0] + ' ' + temp0[1])



    return ans[0], who, best_dressed



ans, who, best_dress = getjoke()


print(ans)
print(who)
print(best_dress)




