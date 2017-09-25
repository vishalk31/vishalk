from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
negative=[]
positive=[]
def create_word_features(words):
    x=words.split(" ")
    useful_words=[]
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