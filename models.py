# models.py

from sentiment_data import *


# Feature extraction base type. Takes an example and returns an indexed list of features.
class FeatureExtractor(object):
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    # Extract features. Includes a flag add_to_indexer to control whether the indexer should be expanded.
    # At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
    def extract_features(self, ex: SentimentExample, add_to_indexer: bool=False):
        raise Exception("Don't call me, call my subclasses")


# Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
class UnigramFeatureExtractor(FeatureExtractor):
    def __init__(self, Indexer: Indexer):
        self.indexer = Indexer
    def get_indexer(self):
        return self.indexer
    def extract_features(self, ex: SentimentExample, add_to_indexer: bool=False):
        idx = [self.indexer.index_of(w) for w in ex.words]
        feat = np.zeros([self.vocab_size()])
        for i in idx:
            feat[i] += 1 
        return feat
    def vocab_size(self):
        return len(self.indexer)


# Bigram feature extractor analogous to the unigram one.
class BigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.get_indexer = Indexer
    def get_indexer(self):
        return self.indexer
    

# Better feature extractor...try whatever techniques you can think of!
class BetterFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


# Sentiment classifier base type
class SentimentClassifier(object):
    # Makes a prediction for the given
    def predict(self, ex: SentimentExample):
        raise Exception("Don't call me, call my subclasses")


# Always predicts the positive class
class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex: SentimentExample):
        return 1


# Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
# superclass
class NaiveBayesClassifier(SentimentClassifier):
    def __init__(self, train_exs, feat_extractor):
        self.feat_extractor = feat_extractor
        # training the nb classfier:
        n = len(train_exs)
        mu = 0
        vocab_size = feat_extractor.vocab_size()
        phi = np.zeros([2,vocab_size]) # two class classifier
        for i in range(n):
            mu += train_exs[i].label
            m = len(train_exs[i].words)
            for j in range(m):
                phi[train_exs[i].label, self.feat_extractor.get_indexer().index_of(train_exs[i].words[j])] += 1
        #import ipdb;ipdb.set_trace()
        phi += 2e-8
        phi /= phi.sum(1)[:,None]
        mu /= n
        self.phi = np.log(phi)
        self.mu  = np.array([1 - mu, mu])
        
    def predict(self, ex: SentimentExample):
        feat = self.feat_extractor.extract_features(ex)
        #import ipdb;ipdb.set_trace()
        prob_0 = np.dot(self.phi[0],feat) + np.log(self.mu[0])
        #print(prob_0)
        prob_1 = np.dot(self.phi[1],feat) + np.log(self.mu[1])
        return float(prob_0 < prob_1)
        
# Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
# superclass
class PerceptronClassifier(SentimentClassifier):
    def __init__(self, train_exs, feat_extractor):
        self.feat_extractor = feat_extractor
        #traning the perceptron classifier
        n = len(train_exs)
        m = self.feat_extractor.vocab_size()
        weight_vector = np.zeros([m])
        Epoch = 10
        for i in range(Epoch):
            acc = np.zeros([n])
            for j in range(n):
                feat = self.feat_extractor.extract_features(train_exs[j])
                pred = float(np.dot(weight_vector,feat) > 0)
                if pred == train_exs[j].label:
                    acc[j] = 1
                    continue
                else:
                    if pred == 0 and train_exs[j].label == 1:
                        weight_vector = weight_vector + feat
                    else:
                        weight_vector = weight_vector - feat
            print('epoch: %s, acc: %.6f' % (i, np.mean(acc)))
        self.weight_vector = weight_vector

    def predict(self, ex: SentimentExample):
        feat = self.feat_extractor.extract_features(ex)
        if np.dot(self.weight_vector, feat) > 0:
            return 1
        else:
            return 0



# Train a Naive Bayes model on the given training examples using the given FeatureExtractor
def train_nb(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> NaiveBayesClassifier:
    model = NaiveBayesClassifier(train_exs, feat_extractor)
    return model


# Train a Perceptron model on the given training examples using the given FeatureExtractor
def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    model = PerceptronClassifier(train_exs, feat_extractor)
    return model

# Main entry point for your modifications. Trains and returns one of several models depending on the args
# passed in from the main method.
def train_model(args, train_exs):
    # Initialize feature extractor
    if args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        n = len(train_exs)
        my_indexer = Indexer()
        for i in range(n):
            #import ipdb;ipdb.set_trace()
            m = len(train_exs[i].words)
            for j in range(m):
                my_indexer.add_and_get_index(train_exs[i].words[j], add=True)

        feat_extractor = UnigramFeatureExtractor(my_indexer)
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "NB":
        model = train_nb(train_exs, feat_extractor)
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, NB, or PERCEPTRON to run the appropriate system")
    return model