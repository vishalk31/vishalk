from nltk.corpus import twitter_samples #importing all the libraries
from nltk.corpus import stopwords #stop word package is used to remove all un-useful words
from nltk.tokenize import word_tokenize#converting text to words can be done using this package
import nltk.classify.util #to find accuracy of our classification
from nltk.classify import NaiveBayesClassifier # naive bayes classifier 
negative=[] #creating positive and negative list to store positive comments and negative comments
positive=[]
def create_word_features(words):# split the text into words
    x=words.split(" ")#splitting passed on the space
    useful_words=[] #list which contains the useful words to process
    useless_words=stopwords.words("english")
    for i in x:
        if(i.isalpha()==True)and i not in useless_words:
            useful_words.append(i)
    my_dict = dict([(use, True) for use in useful_words])
    return my_dict
strings = twitter_samples.strings('negative_tweets.json')
for string in strings:
    negative.append((create_word_features(string),"negative"))
positive_strings = twitter_samples.strings('positive_tweets.json')
for pos in positive_strings:
    positive.append((create_word_features(pos),"positive"))
train_set = negative[:4000] + positive[:4000]
test_set =  negative[4000:] + positive[4000:]
print(len(train_set),  len(test_set))
classifier = NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(accuracy * 100)
review='''im not well'''
words = create_word_features(review)
print classifier.classify(words)
