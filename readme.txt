# update 01/27 -- Jiawei

Mongodb related functions are completed
I find we don't need a json parser because the json format can read directly into mongodb

And I add 2 functions to Utils.py

readJsonIntoDB(filepath, collectionName) can help you read json files into mongodb
The collectionName can be "gg2013" or "gg2015" depending on the data you read

readDBIntoTweetList(collectionName) can help you search db and convert the contents into a list of Tweet
Also you need to specify the collectionName because we will have 2 MongoDB Collection(gg2013 and gg2015)

In order to use those functions, you need to install pymongo, just use pip and follow the instruction here:
https://api.mongodb.com/python/current/installation.html

If you need to install MongoDB, you can follow the instruction:
https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/

Just use the default settings because we will put our db at /data/db
Note that our db name is nlpTeamJJC(Jaieu, Jiawei and Chenbo), not local/config/test/admin
Don't worry, just a note, simply execute readJsonIntoDB() is fine.

If you use pycharm, you can search for a plugin called "Mongo Plugin", a GUI tool for MongoDB