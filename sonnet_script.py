"""
	The goal of this script is to identify the most common term that appears in clusters of similar sonnets
	by Shakespeare using kMeans clustering.

	I ran 100 trials. In each trial, I stored the top 3 terms that appeared in each cluster of that trial.
	Then, I created a words-to-frequency list of all the top terms across all trials and returned the first
	instance of the most frequent term(s).

	In majority of the cases, "love" appeared to be the most frequent occuring term in sonnet clusters.
"""



import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import codecs



class Sonnet:
	"A Sonnet class to store sonnets"

	def __init__(self, text=None):
		""" Creates a new sonnet object.

		Args:
			text (str): sonnet text, empty by default
						Else, text is segmented by space
		"""

		if (text is None):
			self.text = []
		else:
			self.text = text.split()

	def addword(self, word):
		""" Adds word to sonnet text"""
		self.text.append(word)

	def gettext(self):
		""" Returns sonnet text of sonnet"""
		return self.text


def load_sonnets(filename):
    """ Creates sonnet objects from text file.

        Args:
            filename: Testfile of sonnets

        Returns:
            List of sonnet objects

    """

    myfile = codecs.open(filename, "r", "utf-8")
    sonnets = myfile.read().splitlines() # Skip year, title, author
    mysonnets = []

    for line in sonnets:
        # creating first sonnet
        if line == "SONNET" + " " + str(1):
            sonnet = Sonnet()
            nextline = 2
        elif line == "": # skip blanks
            continue
        elif line == "SONNET" + " " + str(nextline):
            # add prev sonnet to list, create next sonnet
            mysonnets.append(sonnet)
            sonnet = Sonnet()
            nextline += 1
        else:
            for w in line.split():
                sonnet.addword(w)

    mysonnets.append(sonnet) # for last sonnet
    return mysonnets



def create_sonnet_corpus(mysonnets):
	"Converts list of sonnet objects to a list of strings"
	corpus = []
	for sonnet in mysonnets:
		corpus.append(' '.join(sonnet.gettext()))
	return corpus


def generate_unique_terms(words_list):
	""""Creates a list of unique terms from a list of words.

	>>> l = ['hi', 'hi', 'hello', 'who', 'dog', 'dog', 'dog', 'cat', 'cat']
	>>> print generate_unique_terms(l)
	['hi', 'hello', 'who', 'dog', 'cat']
	"""

	unique_terms = []
	for w in words_list:
		if w not in unique_terms:
			unique_terms.append(w)
	return unique_terms


def words_to_freq(words_list):
	"""Creates a list of words to frequency mapping.


	>>> l = ['hi', 'hi', 'hello', 'who', 'dog', 'dog', 'dog', 'cat', 'cat']
	>>> print words_to_freq(l)
	[2, 1, 1, 3, 2]
	"""

	unique_terms = generate_unique_terms(words_list)
	words_frequency = [0] * len(unique_terms)

	for w in words_list:
		words_frequency[unique_terms.index(w)] = words_frequency[unique_terms.index(w)] + 1
	return words_frequency

def find_most_common_term(words_list):
	""" Given a list of words, returns the first instance of the most frequently appearing term.

	>>> l = ['hi', 'hi', 'hello', 'who', 'dog', 'dog', 'dog', 'cat', 'cat']
	>>> print find_most_common_term(l)
	dog
	"""

	words_frequency = words_to_freq(words_list)
	unique_terms = generate_unique_terms(words_list)
	max_index = words_frequency.index(max(words_frequency))
	return unique_terms[max_index]







######################################################################
######################### PREPROCESSORS ##############################
######################################################################

def punctuation_removal(mysonnets):
	""" Removes punctuations from sonnets.

		Args:
			mysonnets: list of sonnet objects

		Returns:
			List of sonnet objects

	>>> s = Sonnet("Feed'st thy light's flame with self-substantial fuel,")
	>>> s1 = Sonnet("And tender churl mak'st waste in niggarding:")
	>>> mysonnets = [s, s1]
	>>> result = punctuation_removal(mysonnets)
	>>> print result[0].gettext()
	['Feed', 'thy', 'light', 'flame', 'with', 'self', 'fuel']
	>>> print result[1].gettext()
	['And', 'tender', 'churl', 'mak', 'waste', 'in', 'niggarding']


	"""

	result = []
	for sonnet in mysonnets:
		newsonnet = Sonnet()
		tokenizer = RegexpTokenizer(r'\w+')
		text = sonnet.gettext()
		for word in text:
			token = tokenizer.tokenize(word)
			if token != []:
				newsonnet.addword(token[0])
		result.append(newsonnet)
	return result

def casefold(mysonnets):
	""" Casefolds sonnets.

		Args:
			mysonnets: list of sonnet objects

		Returns:
			List of sonnet objects


	>>> s = Sonnet("Feed thy LiGht flame with Self fuel")
	>>> s1 = Sonnet("And Tender cHurl mak wAste in Niggarding")
	>>> mysonnets = [s, s1]
	>>> result = casefold(mysonnets)
	>>> print result[0].gettext()
	['feed', 'thy', 'light', 'flame', 'with', 'self', 'fuel']
	>>> print result[1].gettext()
	['and', 'tender', 'churl', 'mak', 'waste', 'in', 'niggarding']

	"""
	result = []
	for sonnet in mysonnets:
		newsonnet = Sonnet()
		text = sonnet.gettext()
		for word in text:
			newsonnet.addword(word.lower())
		result.append(newsonnet)
	return result

def filter_shakesperean_words(mysonnets):
    """ Filters sentence of common shakesperean words

		Args:
			mysonnets: list of sonnet objects

		Returns:
			List of sonnet objects


	>>> s1 = Sonnet("as I not for my self but for thee will")
	>>> s2 = Sonnet("thou art the grave where buried love doth live")
	>>> mysonnets = [s1, s2]
	>>> result = filter_shakesperean_words(mysonnets)
	>>> print result[0].gettext()
	['as', 'I', 'not', 'for', 'my', 'self', 'but', 'for', 'will']
	>>> print result[1].gettext()
	['art', 'the', 'grave', 'where', 'buried', 'love', 'live']

	"""

    shakesperean_words = ['thou', 'thy', 'thine', 'thee', 'ye', 'doth', 'dost', 'hath', 'nor', 'th', 'shalt']

    result = []

    for sonnet in mysonnets:
    	newsonnet = Sonnet()
    	text = sonnet.gettext()
    	for word in text:
    		if (word not in shakesperean_words):
    			newsonnet.addword(word)
    	result.append(newsonnet)
    return result


def stopwordremoval(filename, mysonnets):
	""" Removes stopwords from sonnets.

		Args:
			filename: Text file of stopwords
			mysonnets: list of sonnet objects

		Returns:
			List of sonnet objects


	>>> s = Sonnet("Will be a tattered weed of small worth held")
	>>> s1 = Sonnet("Or who is he so fond will be the tomb")
	>>> s2 = Sonnet("So great a sum of sums yet canst not live?")
	>>> mysonnets = [s, s1, s2]
	>>> result = stopwordremoval("stopwords.txt", mysonnets)
	>>> print result[0].gettext()
	['Will', 'tattered', 'weed', 'worth', 'held']
    >>> print result[1].gettext()
    ['Or', 'fond', 'tomb']
	>>> print result[2].gettext()
	['So', 'sum', 'sums', 'canst', 'live?']

	"""

	mystopwords = open(filename, "r")
	stopwords = mystopwords.read().splitlines()

	result = []

	for sonnet in mysonnets:
		newsonnet = Sonnet()
		text = sonnet.gettext()
		for word in text:
			if word not in stopwords:
				newsonnet.addword(word)
		result.append(newsonnet)
	return result



######################################################################
######################################################################
######################################################################




if __name__ == "__main__":
	import doctest
    	doctest.testmod()

	# Load Data
	mysonnets = load_sonnets("sonnets.txt")

	# Prepossing steps
	mysonnets = punctuation_removal(mysonnets)
	mysonnets = casefold(mysonnets)
	mysonnets = stopwordremoval("stopwords.txt", mysonnets)
	mysonnets = filter_shakesperean_words(mysonnets)

	# Create corpus
	corpus = create_sonnet_corpus(mysonnets)

	# Feature extraction
	vectorizer = TfidfVectorizer(min_df=1)
	X = vectorizer.fit_transform(corpus)
	print("n_samples: %d, n_features: %d" % X.shape)
	print "\n"


	# Results
	words_list = []
	n_trials = 100
	for i in range(n_trials):
		# Clustering
		km = KMeans(init='k-means++', max_iter=100, n_init=1)
		km.fit(X)

		order_centroids = km.cluster_centers_.argsort()[:, ::-1]
		terms = vectorizer.get_feature_names()
		for i in range(8): # default amount of clusters is 8
			# Uncomment below lines to see top 3 terms for each cluster
			# Make sure to change n_trials
			# print("Cluster %d:" % i)
			for ind in order_centroids[i, :3]: # adjust to display # of top terms
				#print(' %s' % terms[ind])
				words_list.append(terms[ind])

	print "# of trials: ", n_trials
	print "Most common term: ", find_most_common_term(words_list)









