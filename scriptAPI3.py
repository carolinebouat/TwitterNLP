# Importing useful packages 

from tweepy.streaming import Stream
import json
import time
import sys
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy import API
from tweepy import StreamListener

#Identification 

API_key = 'NpiKR3X8LLuCcjbCDWzmHxk1O'
API_keysecret = 'p1bG1GxGYwkAopJzwQF11BrnMhvGZejfhTevN08MrMATsnMNA1'

Access_token = '1465024506091687936-9VHr5sjMusEouQYfAvinJlVa0YwbNo'
Access_token_secret = 'w22BsKo6wPZ9ba22BFmoXOahi739stjk8wf5SfvQkVmiZ'

auth = OAuthHandler(API_key, API_keysecret)
auth.set_access_token(Access_token, Access_token_secret)
api = API(auth)

#Definition of Twitter stream listener Class 

class SListener(StreamListener):
    def __init__(self, api = None, fprefix = 'streamer'):
        self.api = api or API()
        self.counter = 0
        self.fprefix = fprefix
        self.output  = open('%s_%s.json' % (self.fprefix, time.strftime('%Y%m%d-%H%M%S')), 'w')


    def on_data(self, data):
        print("Found one")
        if  'in_reply_to_status' in data:
            self.on_status(data)
        elif 'delete' in data:
            delete = json.loads(data)['delete']['status']
            if self.on_delete(delete['id'], delete['user_id']) is False:
                return False
        elif 'limit' in data:
            if self.on_limit(json.loads(data)['limit']['track']) is False:
                return False
        elif 'warning' in data:
            warning = json.loads(data)['warnings']
            print("WARNING: %s" % warning['message'])
            return


    def on_status(self, status):
        print(status)
        self.output.write(status)
        self.counter += 1
        if self.counter >= 2000:
            self.output.close()
            self.output  = open('%s_%s.json' % (self.fprefix, time.strftime('%Y%m%d-%H%M%S')), 'w')
            self.counter = 0
        return


    def on_delete(self, status_id, user_id):
        print("Delete notice")
        return

    def on_limit(self, track):
        print("WARNING: Limitation notice received, tweets missed: %d" % track)
        return

    def on_error(self, status_code):
        print('Encountered error with status code:', status_code)
        return 

    def on_timeout(self):
        print("Timeout, sleeping for 60 seconds...")
        time.sleep(60)
        return 

# Set streaming rules  
language = ["fr"]
		
#instantiate the SListener object 
listen = SListener(api)

# instantiate the Stream object 
stream = Stream(auth, listen)

# Collecting data 
#stream.filter(locations=[-4.9,   46.47,   0.04,    49.8]) # Bretagne
stream.filter(locations=[4.82,    43.14,   7.1, 44.28]) # Bouches du Rhone
#stream.filter(locations=[-4.8,    42.46,   8.17,    51.29]) # Toute la France




