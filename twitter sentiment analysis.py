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
    useless_words=stopwords.words("english")#give the stop words
    for i in x:
        if(i.isalpha()==True)and i not in useless_words:
            useful_words.append(i)
    my_dict = dict([(use, True) for use in useful_words])
    return my_dict
strings = twitter_samples.strings('negative_tweets.json')#gets the data of negative commands
for string in strings:
    negative.append((create_word_features(string),"negative")) #call the create_word_features and process the negative commands
positive_strings = twitter_samples.strings('positive_tweets.json')#gets the data of positive commands
for pos in positive_strings:
    positive.append((create_word_features(pos),"positive")) #call the create_word_features and process the positive commands
train_set = negative[:4000] + positive[:4000]# training data 
test_set =  negative[4000:] + positive[4000:]# testing data
print(len(train_set),  len(test_set))
classifier = NaiveBayesClassifier.train(train_set)# classifying with naivebayes
accuracy = nltk.classify.util.accuracy(classifier, test_set)#find the accuracy 
print(accuracy * 100)#output of the accuracy
review='''im not well'''#checking
words = create_word_features(review)
print classifier.classify(words)#classification
