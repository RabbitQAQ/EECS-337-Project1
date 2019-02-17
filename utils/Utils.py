import json
import csv
import re

from pymongo import MongoClient
from difflib import SequenceMatcher
from model.Tweet import Tweet
from model.User import User

# Read local json files into db
def readJsonIntoDB(filepath, collectionName):
    try:
        with open(filepath) as input_file:
            data = json.load(input_file)
            client = MongoClient()
            db = client["nlpTeamJJC"]
            collection = db[collectionName]
            collection.insert(data)
    except:
        print("Read Json Into DB: Operation Failed(" + collectionName + ")")
    else:
        print("Read Json Into DB: Operation Succeed(" + collectionName + ")")

# Return a list of Tweet read from db
def readDBIntoTweetList(collectionName):
    res = []
    client = MongoClient()
    db = client["nlpTeamJJC"]
    collection = db[collectionName]
    # Get cursor from db to get all contents
    cursor = collection.find({})
    # Convert json-like contents to Tweet model
    for item in cursor:
        tmpTweet = Tweet(item['text'], item['timestamp_ms'], item['user']['id'], item['user']['screen_name'],
                         item['id'])
        res.append(tmpTweet)

    # Return a list of Tweet
    return res

def readJsonIntoTweetList(filepath):
    res = []
    cnt = 0
    try:
        with open(filepath) as jsonfile:
            data = json.loads(jsonfile.read())
            for item in data:
                cnt += 1
                print(cnt)
                tmpTweet = Tweet(re.sub("[^a-zA-Z0-9-, ]", "", item['text']), item['timestamp_ms'], item['user']['id'], item['user']['screen_name'],
                                 item['id'])
                res.append(tmpTweet)
    except:
        print("Read Json Into List: Operation Failed(" + filepath+ ")")
    else:
        print("Read Json Into List: Operation Succeed(" + filepath + ")")

    # Return a list of Tweet
    return res

def readJsonIntoTweetListToString(filepath):
    res = []
    cnt = 0
    try:
        with open(filepath) as jsonfile:
            data = json.loads(jsonfile.read())
            for item in data:
                cnt += 1
                print(cnt)
                res.append(re.sub("[^a-zA-Z0-9-, ]", "", item['text']))
    except:
        print("Read Json Into List: Operation Failed(" + filepath+ ")")
    else:
        print("Read Json Into List: Operation Succeed(" + filepath + ")")

    # Return a list of Tweet
    return res

# Return a list of Tweet read from db
def readDBIntoTweetListToString(collectionName):
    res = []
    client = MongoClient()
    db = client["nlpTeamJJC"]
    collection = db[collectionName]
    # Get cursor from db to get all contents
    cursor = collection.find({})
    # Convert json-like contents to string list
    res = []
    for item in cursor:
        res.append(item['text'])

    # Return a list of Tweet
    return res

def tsv_parser(filepath):
    prevName = ''
    with open('../data/new_name.txt', 'a') as newfile:
        with open(filepath) as tsvfile:
            reader = csv.DictReader(tsvfile, dialect='excel-tab')
            for row in reader:
                deathYear = row['deathYear']
                if deathYear == '\\N' and row['primaryName'] != prevName:
                    newfile.write(row['primaryName'] + '\n')
                    prevName = row['primaryName']

def find_name(name):
    res = []
    with open('../data/new_name.txt') as namefile:
        lines = namefile.read().split('\n')
        for line in lines:
            if name in line:
                res.append(line)

    return res

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_most_similar(name):
    mostSim = 0
    mostSimName = ''
    str = name.split(' ')
    for n in find_name(str[0]):
        similarity = similar(n, name)
        if similarity > mostSim:
            mostSim = similarity
            mostSimName = n

    return mostSimName

if __name__ == '__main__':
    # Execute the following line only ONCE for each json file.
    # If you are importing gg2013.json, the second parameter should be gg2013.
    # If importing gg2015.json, then please change it to gg2015.
    # gg2013 is much smaller and suitable for testing.
    #readJsonIntoDB('../data/gg2015.json', "gg2015")

    # An example for read contents from db
    #tweetList = readDBIntoTweetList("gg2013")

    #tsv_parser('../data/name.basics.tsv')

    #print(find_most_similar('Dssd Edasdas'))
    readJsonIntoTweetListToString('../data/gg2015.json')
    print("fuck")
