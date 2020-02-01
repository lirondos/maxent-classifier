# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from collections import defaultdict
import numpy as np
import math
from random import shuffle
import gc
from corpus import Document, NamesCorpus, ReviewCorpus
from random import shuffle, seed
import sys
import matplotlib.pyplot as plt


class MaxEnt(Classifier):

    def get_model(self): return None

    def set_model(self, model): pass

    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        self.train_sgd(instances, dev_instances, 0.0001, 30)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size, my_lambda=0):
        """Train MaxEnt model with Mini-batch Stochastic Gradient
        """
        self.my_lambda = my_lambda
        self.train_instances = train_instances
        self.dev_instances = dev_instances
        self.vocab, self.labels = self.get_vocab()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.featurize(self.train_instances)
        self.parameters = np.zeros((len(self.labels.keys()), len(self.vocab.keys())))
        self.minibatch_grad_desc()

    def minibatch_grad_desc(self):
        #shuffle(self.train_instances)
        done = False
        iterations = 0
        old_nll = 20000
        correct = []
        minibatches = [self.train_instances[i * self.batch_size:(i + 1) * self.batch_size] for i in
                       range((len(self.train_instances) + self.batch_size - 1) // self.batch_size)]
        print("Number of minibatches " + str(len(minibatches)))
        while not done:
            iterations = iterations + 1
            nll = []
            for i in range(len(minibatches)):
                if not done:
                    minibatch = minibatches[i]
                    gradient = np.full_like(self.parameters, 0)
                    for instance in minibatch:
                        nll.append(-math.log(self.get_posterior(instance, instance.label)))
                        gradient = gradient + self.gradient_instance(instance)
                    old_parameters = self.parameters
                    self.parameters -= gradient*self.learning_rate
            correct.append(len([self.classify(x) for x in self.dev_instances if self.classify(x)== x.label]))

            if iterations < 50 or old_nll - np.sum(nll) > 30 :
                old_nll = np.sum(nll)
                print("After the entire training set the NLL is " + str(old_nll))
                gc.collect()
            else:
                done = True


    def gradient_instance(self, instance):
        gd = np.full_like(self.parameters, 0)
        for label, id in self.labels.items():
            posterior = self.get_posterior(instance, label)
            if label == instance.label:
                increment = -1 + posterior
            else:
                increment = -0 + posterior
            gd[id, :] = self.my_lambda*(self.parameters[id]*instance.feature_vector) + instance.feature_vector*increment
        return gd

    def get_posterior(self, instance, label):
        numerator = math.exp(np.dot(self.parameters[self.labels[label]], instance.feature_vector))
        denominator = self.get_denominator(instance)
        return numerator/denominator

    def get_denominator(self, instance):
        denominator = 0
        for label, id in self.labels.items():
            denominator = denominator + math.exp(np.dot(self.parameters[self.labels[label]], instance.feature_vector))
        return denominator


    def featurize(self, instances):
        for instance in instances:
            self.featurize_instance(instance)
            #print(train_instance.features)

    def featurize_instance(self, instance):
        #gc.collect()
        my_featurization = np.zeros(len(self.vocab.items()))
        for feature in instance.features():
            if feature in self.vocab.keys():
                my_featurization[self.vocab[feature]] = my_featurization[self.vocab[feature]] + 1
        my_featurization[self.vocab["_bias_"]] = 1
        instance.feature_vector = my_featurization

    def featurize_for_class(self, feature, my_label):
        """
        Returns a feature representation with 0 everywhere except for the section we are representing
        :param feature: vector representation of an instance (no bias, no sections)
        :param labels_dict:
        :param my_label: the label whose class we want to represent
        :return:
        """
        #print(feature)
        my_array = np.zeros((len(self.vocab.items())*len(self.labels.items())))
        for i in range(len(self.vocab.items())):
            my_array[i*(my_label + 1)] = feature[i]
        my_array[self.vocab["_bias_"]] = 1 #bias
        return my_array


    def get_vocab(self):
        vocab = set()
        labels = set()
        for train_instance in self.train_instances:
            vocab.update(train_instance.features())
            labels.add(train_instance.label)
        vocab = list(vocab)
        vocab.append("_bias_")
        vocab_to_dict = self.iterable_to_dict(vocab)
        label_to_dict = self.iterable_to_dict(labels)
        return vocab_to_dict, label_to_dict

    def iterable_to_dict(self, iterable):
        i = 0
        interable_dict = dict()
        for elem in iterable:
            interable_dict[elem] = i
            i = i + 1
        return interable_dict

    def classify(self, instance):
        self.featurize_instance(instance)
        probs = []
        for label, id in self.labels.items():
            posterior = self.get_posterior(instance, label)
            probs.append(posterior)
        index = np.argmax(probs)
        for label, id in self.labels.items():
            if index == id:
                my_label = label
                return my_label

    def accuracy(self, test):
        correct = [self.classify(x) == x.label for x in test]
        return float(sum(correct)) / len(correct)

class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        return self.data.split()

class Name(Document):
    def features(self):
        name = self.data
        return ['First=%s' % name[0], 'Last=%s' % name[-1]]

def split_review_corpus(reviews, training_size):
    #"Split the yelp review corpus into training, dev, and test sets
    return (reviews[:training_size], reviews[-5000:-4000], reviews[-4000:])


def get_corpus(document_class):
    reviews = ReviewCorpus('yelp_reviews.json', document_class=document_class)
    seed(hash("reviews"))
    shuffle(reviews)
    return reviews

def experiment1(training_sizes):
    reviews = get_corpus(BagOfWords)
    results = []
    for size in training_sizes:
        train, dev, test = split_review_corpus(reviews, size)
        classifier = MaxEnt()
        classifier.train(train, dev)
        accuracy = classifier.accuracy(test)
        print("With training size " + str(size) + " accuracy was " + str(accuracy))
        results.append(accuracy)
    return results

def experiment2(batch_sizes):
    reviews = get_corpus(BagOfWords)
    results = []
    for batch_size in batch_sizes:
        train, dev, test = split_review_corpus(reviews, 10000)
        classifier = MaxEnt()
        classifier.train_sgd(train, dev, 0.0001, batch_size)
        accuracy = classifier.accuracy(dev)
        print("With batch size " + str(batch_size) + " accuracy was " + str(accuracy))
        results.append(accuracy)
    return results

def experiment3(lambda_values):
    reviews = get_corpus(BagOfWords)
    results = []
    for my_lambda in lambda_values:
        train, dev, test = split_review_corpus(reviews, 10000)
        classifier = MaxEnt()
        classifier.train_sgd(train, dev, 0.0001, 10, my_lambda)
        accuracy = classifier.accuracy(dev)
        print("With lambda " + str(my_lambda) + " accuracy was " + str(accuracy))
        results.append(accuracy)
    return results

if __name__== "__main__":

    training_sizes = [1000, 10000, 50000, 100000, -5000]
    results1 = experiment1(training_sizes)
    print(results1)

    batch_sizes = [1, 10, 50, 100, 1000]
    results2 = experiment2(batch_sizes)
    print(results2)

    lambdas = [0.1, 0.5, 1, 10]
    results3 = experiment3(lambdas)
    print(results3)



