'''Version 0.35'''
# Imports
import re
import os
import string

import nltk
import json
import numpy as np
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from utils import Utils
from utils.Utils import readDBIntoTweetList, readDBIntoTweetListToString
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from imdb import IMDb

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
        cnt += 1
        print(cnt)
        sentences = sentDetector.tokenize(tweet.get_text())
        if not retweetCleanerRE.search(sentences[0]):
            for s in sentences:
                if keywordsCleanerRE.search(s):
                    cleanedTweet = re.sub("[^a-zA-Z0-9- ]", "", tweet.get_text())
                    cleanedTweetList.append(cleanedTweet)
                break

    return cleanedTweetList

def findwinner(cleanedTweetList, lines, i, word_tfidf, weight):
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
    awardWords = [w for w in tweetTokenizer.tokenize(awardstring) if not w in stopwordlist]
    categorywords = [w for w in tweetTokenizer.tokenize(catagorystring) if not w in stopwordlist]
    if len(categorywords) == 0:
        categorywords.append(" ")
    words = awardWords + categorywords

    sortedDict = {}

    if len(awardWords) >= 8:
        sortedDict = findWinnerInNgrams(cleanedTweetList, i, awardWords, categorywords, word_tfidf, weight, 3)

    if len(awardWords) < 8 or len(sortedDict) == 0:
        sortedDict = findWinnerInNgrams(cleanedTweetList, i, awardWords, categorywords, word_tfidf, weight, 2)

    if len(sortedDict) == 0:
        sortedDict = findWinnerInNgrams(cleanedTweetList, i, awardWords, categorywords, word_tfidf, weight, 1)


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


def findWinnerInNgrams(cleanedTweetList, i, awardWords, categoryWords, word_tfidf, weight, n):
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
    return sortedDict

# Autograder
OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']

def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    return []

def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    return []

def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''
    # Your code here
    res = {}
    if year == '2013' or year == '2015':
        for award in OFFICIAL_AWARDS_1315:
            res[award] = ''
    if year == '2018' or year == '2019':
        for award in OFFICIAL_AWARDS_1819:
            res[award] = ''
    return res

def get_winner(year):
    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    ############################################# PREPROCESS
    # TF-IDF
    lines = []
    if year == '2013' or year == '2015':
        lines = OFFICIAL_AWARDS_1315[:]
    if year == '2018' or year == '2019':
        lines = OFFICIAL_AWARDS_1819[:]
    corpus = lines
    # Word to frequency matrix
    vectorizer = CountVectorizer()
    # Calculate the times a word appears
    X = vectorizer.fit_transform(corpus)
    # Get every key word
    word_tfidf = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    # Calculate TF-IDF
    tfidf = transformer.fit_transform(X)
    weight = tfidf.toarray()
    cleanedTweets = tweetsCleaner(readDBIntoTweetList('gg2013'))
    ############################################ PREPROCESS END
    res = {}
    # for award in OFFICIAL_AWARDS_1315:
    #     res[award] = ''
    if year == '2013':
        for i in range(0, len(lines)):
            res[lines[i]] = string.capwords(findwinner(cleanedTweets, lines, i, word_tfidf, weight))
    return res

def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    # Your code here
    res = {}
    if year == '2013' or year == '2015':
        for award in OFFICIAL_AWARDS_1315:
            res[award] = ''
    if year == '2018' or year == '2019':
        for award in OFFICIAL_AWARDS_1819:
            res[award] = ''
    return res

def pre_ceremony():
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    # Clean Tweet
    global cleanedTweetList
    global word_tfidf
    global weight
    cleanedTweetList = tweetsCleaner(readDBIntoTweetList("gg2013"))
    # TF-IDF
    file = open(datapath + "/AwardCategories2013.txt")
    lines = file.read().split("\n")
    corpus = lines
    # Word to frequency matrix
    vectorizer = CountVectorizer()
    # Calculate the times a word appears
    X = vectorizer.fit_transform(corpus)
    # Get every key word
    word_tfidf = vectorizer.get_feature_names()

    transformer = TfidfTransformer()

    # Calculate TF-IDF
    tfidf = transformer.fit_transform(X)
    weight = tfidf.toarray()

    for wi in range(0, len(weight[0])):
        count = 0
        for wj in range(0, len(weight)):
            if weight[wj][wi] != 0:
                count += 1
                temp = wj
        if count == 1:
            weight[temp][wi] *= 10
    print("Pre-ceremony processing complete.")
    return

def main():
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''
    # Your code here
    pre_ceremony()
    return

if __name__ == '__main__':
    main()


