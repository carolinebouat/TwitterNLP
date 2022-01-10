def contains(original, targets):
	'''
	check wether the is a least one occurence of each of the words in 
	targets in original. This is not case-dependent.
	'''
	return any(s in original.lower() for s.lower() in targets)
