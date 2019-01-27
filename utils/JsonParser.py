import json

from model.Tweet import Tweet
from model.User import User


def jsonParser(filename):
    res = []
    with open(filename) as input_file:
        data = json.load(input_file)
        for item in data:
            tmpTweet = Tweet(item['text'], item['timestamp_ms'], item['user']['id'], item['user']['screen_name'], item['id'])
            res.append(tmpTweet)

    return res


if __name__ == '__main__':
    # example
    tweetList = jsonParser('../data/gg2013.json')
    print(tweetList)
