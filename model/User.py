class User:

    def __init__(self, input_id, input_screen_name):
        self.id = input_id
        self.screen_name = input_screen_name

    def get_id(self):
        return self.id

    def get_screen_name(self):
        return self.screen_name

    # def set_id(self, input_id):
    #     self.id = input_id
    #
    # def set_screen_name(self, input_screen_name):
    #     self.id = input_screen_name
