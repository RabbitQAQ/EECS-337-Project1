from model.User import User


class Twitter:
    def __init__(self, input_text, input_timestamp_ms, input_id, input_sceen_name, input_tweet_id):
        self.text = input_text
        self.timestamp_ms = input_timestamp_ms
        self.user = User(input_id, input_sceen_name)
        self.id = input_tweet_id

    def get_text(self):
        return self.text

    def get_timestamp_ms(self):
        return self.timestamp_ms

    def get_user(self):
        return self.user

    def get_id(self):
        return self.id

    #comment test
