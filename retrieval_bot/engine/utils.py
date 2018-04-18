import random
import os 

class DefaultMessage:

    def __init__(self, paths=None):
        self.messages = set()
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if not paths and os.path.exists('{}/default_messages.txt'.format(dir_path)):
            paths = ['{}/default_messages.txt'.format(dir_path)]

        if paths:
            try:
                for path in paths:
                    with open(path, encoding='utf-8') as f:
                        self.messages.update({line.strip() for line in f})
            except Exception as e:
                print(e)

    def get_random_message(self):
        if self.messages:
            return random.sample(self.messages, 1)[0]
        return "I don't know what I have to say"