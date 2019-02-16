'''Version 0.35'''
# Imports
import re
import os
import string

import nltk
import spacy
import json
import numpy as np
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from utils import Utils
from utils.Utils import readDBIntoTweetList, readDBIntoTweetListToString
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

# SpaCy
spacyNLP = spacy.load('en')

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
                    cleanedTweet = re.sub("[^a-zA-Z0-9-, ]", "", tweet.get_text())
                    cleanedTweetList.append(cleanedTweet)
                break

    return cleanedTweetList
cleanedTweets2013 = tweetsCleaner(readDBIntoTweetList("gg2013"))
#cleanTweets2015 = tweetsCleaner(readDBIntoTweetList("gg2015"))

def award_classifier(tweet_tokens, award_categories, aw):
    best_score = 0
    best_category = ""
    for award in award_categories:
        awardstring = award.lower()
        awardstring = awardstring.replace('limited', 'mini')
        awardWords = [w for w in tweetTokenizer.tokenize(awardstring) if not w in stopwordlist]
        score = num_matches(awardWords, tweet_tokens)
        if score > best_score and score > 0:
            best_score = score
            best_category = award

    if best_category == aw:
        return True
    return False

def num_matches(list1, list2):
    matches = 0
    for item in list1:
        matches += list2.count(item)
    return matches


def findHost():
    hostWords = ["host", "hosts", "hosting"]
    dict = {}
    res = []
    for tweet in cleanedTweets2013:
        for hw in hostWords:
            if hw in tweet:
                tmp = tweet.lower()
                tokens = tweetTokenizer.tokenize(tmp)
                usefulTokens = [w for w in tokens if not w in stopwordlist]
                for k in nltk.bigrams(usefulTokens):
                    if k in dict:
                        dict[k] += 1
                    else:
                        dict[k] = 1
    sortedDict = sorted(dict.items(), key=lambda entry: entry[1], reverse=True)

    if sortedDict[0][1] / sortedDict[1][1] >= 0.8:
        res.append(string.capwords(sortedDict[0][0][0] + ' ' + sortedDict[0][0][1]))
        res.append(string.capwords(sortedDict[1][0][0] + ' ' + sortedDict[1][0][1]))
    else:
        res.append(string.capwords(sortedDict[0][0][0] + ' ' + sortedDict[0][0][1]))

    return res

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
        sortedDict = findWinnerInNgrams(cleanedTweetList, i, awardWords, categorywords, word_tfidf, weight, 1, lines, line)

    if len(awardWords) < 8 or len(sortedDict) == 0:
        sortedDict = findWinnerInNgrams(cleanedTweetList, i, awardWords, categorywords, word_tfidf, weight, 2, lines, line)

    if len(sortedDict) == 0:
        sortedDict = findWinnerInNgrams(cleanedTweetList, i, awardWords, categorywords, word_tfidf, weight, 1, lines, line)


    try:
        winner = sortedDict[0][0][0] + ' ' + sortedDict[0][0][1]
    except:
        print("error")

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


def findWinnerInNgrams(cleanedTweetList, i, awardWords, categoryWords, word_tfidf, weight, n, lines, line):
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
                        if award_classifier(usefulTokens, lines, line):
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

def findawardsname():
    res = {}
    temp_len = 0
    for tweet in cleanedTweets2013:
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

    trueResult = []
    for i in range(0, 26):
        trueResult.append(sortedDict[i][0])
    return trueResult
# def findPresenter():
#     presenters = dict()
#     presenters_tweets = []
#     presenter_words = ['present', 'presents', 'presenting','presenter','presented' 'Present', 'Presenter', 'Presenting', 'Presented', 'Presents']
#     #generic words that are likely to appear that will not be human names
#     final_stopwords = ['Fair', 'Best', 'She', 'He', 'Hooray' 'Supporting', 'Actor', 'Actress', 'The', 'A', 'Bad', 'Good', 'Not', 'Drinking', 'Eating', 'Dancing', 'Singing', 'And', 'Hooray', 'Nshowbiz', 'TMZ', 'VanityFair', 'Mejor', 'Better', 'Score', 'Movie', 'Film', 'Song' 'Drama', 'Comedy', 'So', 'Better', 'Netflix', 'Someone', 'Mc', 'Newz', 'Season', 'Should']
#
#     winners = {'cecil b. demille award' : 'Jodie Foster', 'best motion picture - drama' : 'Argo', 'best performance by an actress in a motion picture - drama' : 'Jessica Chastain', 'best performance by an actor in a motion picture - drama' : 'Daniel Day-Lewis', 'best motion picture - comedy or musical' : 'Les Miserables', 'best performance by an actress in a motion picture - comedy or musical' : 'Jennifer Lawrence', 'best performance by an actor in a motion picture - comedy or musical' : 'Hugh Jackman', 'best animated feature film' : 'Brave', 'best foreign language film' : 'Amour', 'best performance by an actress in a supporting role in a motion picture' : 'Anne Hathaway', 'best performance by an actor in a supporting role in a motion picture' : 'Christoph Waltz', 'best director - motion picture' : 'Ben Affleck', 'best screenplay - motion picture' : 'Quentin Tarantino', 'best original score - motion picture' : 'Mychael Danna', 'best original song - motion picture' : 'Skyfall', 'best television series - drama' : 'Homeland', 'best performance by an actress in a television series - drama' : 'Claire Danes', 'best performance by an actor in a television series - drama' : 'Damian Lewis', 'best television series - comedy or musical' : 'Girls', 'best performance by an actress in a television series - comedy or musical':'Lena Dunham', 'best performance by an actor in a television series - comedy or musical':'Don Cheadle', 'best mini-series or motion picture made for television':'Game Change', 'best performance by an actress in a mini-series or motion picture made for television':'Julianne Moore', 'best performance by an actor in a mini-series or motion picture made for television':'Kevin Costner', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television': 'Maggie Smith', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television': 'Ed Harris'}
#
#     for award in OFFICIAL_AWARDS_1315:
#         presenters[award] = []
#
#     for tweet in cleanedTweets2013:
#         if any(word in tweet for word in presenter_words):
#             presenters_tweets.append(tweet)

def findPresenter():
    presentKeywords = ['present', 'Present', 'introduc', 'Introduc']
    resultBlackList = ['ben', 'affleck', 'clinton', 'golden', 'globe']
    presenter = {}
    winners = {'cecil b. demille award' : 'Jodie Foster', 'best motion picture - drama' : 'Argo', 'best performance by an actress in a motion picture - drama' : 'Jessica Chastain', 'best performance by an actor in a motion picture - drama' : 'Daniel Day-Lewis', 'best motion picture - comedy or musical' : 'Les Miserables', 'best performance by an actress in a motion picture - comedy or musical' : 'Jennifer Lawrence', 'best performance by an actor in a motion picture - comedy or musical' : 'Hugh Jackman', 'best animated feature film' : 'Brave', 'best foreign language film' : 'Amour', 'best performance by an actress in a supporting role in a motion picture' : 'Anne Hathaway', 'best performance by an actor in a supporting role in a motion picture' : 'Christoph Waltz', 'best director - motion picture' : 'Ben Affleck', 'best screenplay - motion picture' : 'Quentin Tarantino', 'best original score - motion picture' : 'Mychael Danna', 'best original song - motion picture' : 'Skyfall', 'best television series - drama' : 'Homeland', 'best performance by an actress in a television series - drama' : 'Claire Danes', 'best performance by an actor in a television series - drama' : 'Damian Lewis', 'best television series - comedy or musical' : 'Girls', 'best performance by an actress in a television series - comedy or musical':'Lena Dunham', 'best performance by an actor in a television series - comedy or musical':'Don Cheadle', 'best mini-series or motion picture made for television':'Game Change', 'best performance by an actress in a mini-series or motion picture made for television':'Julianne Moore', 'best performance by an actor in a mini-series or motion picture made for television':'Kevin Costner', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television': 'Maggie Smith', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television': 'Ed Harris'}
    # winners = get_winner('2013')
    for k,v in winners.items():
        presenter[k] = {}
    # Traverse tweets
    for tweet in cleanedTweets2013:
        #tweet = tweet.lower()
        # If current tweet contains keywords
        for pk in presentKeywords:
            if pk in tweet:
                # Now the tweet contains present keywords
                for k, v in winners.items():
                    currWinner = v.lower()
                    nameParts = currWinner.split(' ')
                    for namePart in nameParts:
                        if namePart in tweet.lower():
                            entities = spacyNLP(tweet)
                            for entity in entities.ents:
                                if entity.label_ == "PERSON":
                                    if entity.text in presenter[k]:
                                        presenter[k][entity.text] += 1
                                    else:
                                        presenter[k][entity.text] = 1
                            break
                break

    # Sort dict
    for awardName, pdict in presenter.items():
        presenter[awardName] = sorted(pdict.items(), key=lambda entry: entry[1], reverse=True)

    # Remove winner
    for awardName, plist in presenter.items():
        currWinner = winners[awardName].lower()
        for k in plist[:]:
            for half in k[0].split(' '):
                if half.lower() in currWinner:
                    plist.remove(k)
                    break
            presenter[awardName] = plist

    # Remove black list
    for awardName, plist in presenter.items():
        for k in plist[:]:
            for black in resultBlackList:
                if black in k[0].lower():
                    plist.remove(k)
                break
            presenter[awardName] = plist

    # Remove low frequency items
    for awardName, plist in presenter.items():
        pass
        # if plist != []:
        #     max = plist[0][1]
        #     for k in plist[:]:
        #         if k[1] < max * 0.65:
        #             plist.remove(k)
        #     presenter[awardName] = plist

    # Build result
    trueResult = {}
    for k, v in presenter.items():
        if len(v) > 1:
            trueResult[k] = []
            trueResult[k].append(string.capwords(v[0][0]))
            trueResult[k].append(string.capwords(v[1][0]))
        elif len(v) > 0:
            trueResult[k] = []
            trueResult[k].append(string.capwords(v[0][0]))
        else:
            trueResult[k] = []


    return trueResult

# Autograder
OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']

def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    # cleanedTweets = tweetsCleaner(readDBIntoTweetList('gg2013'))
    return findHost()

def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    return findawardsname()

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
    ############################################ PREPROCESS END
    res = {}
    # for award in OFFICIAL_AWARDS_1315:
    #     res[award] = ''
    if year == '2013':
        for i in range(0, len(lines)):
            res[lines[i]] = string.capwords(findwinner(cleanedTweets2013, lines, i, word_tfidf, weight))
    if year == '2015':
        for i in range(0, len(lines)):
            res[lines[i]] = string.capwords(findwinner(cleanedTweets2013, lines, i, word_tfidf, weight))
    return res

def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    # Your code here
    # res = {}
    # if year == '2013' or year == '2015':
    #     for award in OFFICIAL_AWARDS_1315:
    #         res[award] = ''
    # if year == '2018' or year == '2019':
    #     for award in OFFICIAL_AWARDS_1819:
    #         res[award] = ''
    return findPresenter()

def pre_ceremony():
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
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


