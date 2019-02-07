import json

from pymongo import MongoClient

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
        print("Read Json Into DB: Operation Failed")
    else:
        print("Read Json Into DB: Operation Succeed")

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


if __name__ == '__main__':
    # Execute the following line only ONCE for each json file.
    # If you are importing gg2013.json, the second parameter should be gg2013.
    # If importing gg2015.json, then please change it to gg2015.
    # gg2013 is much smaller and suitable for testing.
    readJsonIntoDB('../data/gg2015.json', "gg2015")

    # An example for read contents from db
    #tweetList = readDBIntoTweetList("gg2013")
