from discord import Webhook, RequestsWebhookAdapter
import requests
import json
import os

class ProgBar():
    def __init__(self, prefix='', msg='', val=0):
        self._prefix = prefix
        self._msg = msg
        self._val = val

        # Gets webhook if it exists.
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                self._webhook = Webhook.from_url(json.load(f)['webhook'], adapter=RequestsWebhookAdapter())
        else:
            self._webhook = None

    def set_prefix(self, prefix):
        self._prefix = prefix

    def get_prefix(self):
        return self._prefix

    def set_msg(self, msg):
        self._msg = msg

    def get_msg(self):
        return self._msg

    def set_val(self, val):
        self._val = val
        percent = ("{0:." + str(1) + "f}").format(100*val)
        length = os.get_terminal_size().columns - len(self._prefix) - 20
        filledLength = int(length*val)
        bar = '█' * filledLength + '-' * (length - filledLength)
        print(f'\r{self._prefix} |{bar}| {percent}% Complete', end='\r')

        if val == 1:
            print()

            if self._msg != '' and self._webhook is not None:
                self._webhook.send(self._msg)

    def get_val(self):
        return self._val

    # Uses getters and setters.
    prefix = property(get_prefix, set_prefix)
    msg = property(get_msg, set_msg)
    val = property(get_val, set_val)
    