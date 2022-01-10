import json
from glob import glob
import pickle


tweets = []
for file in glob("*.json"):
	file = open(file)
	for line in file.readlines():
		try:
			tweets.append(json.loads(line))
		except :
			print("We skiped a tweet")


pickle.dump(tweets, open("all_combined.pickle", "wb"))


def contains(original, targets):
	'''
	check wether the is a least one occurence of each of the words in 
	targets in original. This is not case-dependent.
	'''
	return any(s.lower() in original.lower() for s in targets)