import numpy as np
import math
import glob
import re
from typing import List


class wordCounter():
    '''
    Placeholder to store information about each entry (word) in a dictionary
    '''

    def __init__(self, word, numOfHamWords, numOfSpamWords, p):
        self.word = word
        self.numOfHamWords = numOfHamWords
        self.numOfSpamWords = numOfSpamWords
        self.p = p


class naiveBayes():
    '''
    Naive bayes class
    Train model and classify new emails
    '''

    def _extractWords(self, filecontent: str) -> List[str]:
        '''
        Word extractor from filecontent
        :param filecontent: filecontent as a string
        :return: list of words found in the file
        '''
        txt = filecontent.split(" ")
        txtClean = [(re.sub(r'[^a-zA-Z]+', '', i).lower()) for i in txt]
        words = [i for i in txtClean if i.isalpha()]
        return words

    def train(self, msgDirectory: str, fileFormat: str = '*.txt') -> (List[wordCounter], float):
        '''
        :param msgDirectory: Directory to email files that should be used to train the model
        :return: model dictionary and model prior
        '''
        files = sorted(glob.glob(msgDirectory + fileFormat))
        # TODO: Train the naive bayes classifier
        # TODO: Hint - store the dictionary as a list of 'wordCounter' objects
        ham_words = []
        spam_words = []
        final_dictionary = []
        spam_count = 0
        processed_words = []

        for i in range(len(files)):
            f = open(files[i], 'r')
            if 'spm' in files[i]:
                spam_words.extend(self._extractWords(f.read()))
                spam_count += 1
            else:
                ham_words.extend(self._extractWords(f.read()))
        ham_amount = len(ham_words)
        spam_amount = len(spam_words)
        total_amount = ham_amount + spam_amount
        for i in range(ham_amount):
            current_word = ham_words[i]
            self.add_new_word(current_word, final_dictionary, ham_words, processed_words, spam_words, total_amount)
        for i in range(spam_amount):
            current_word = spam_words[i]
            self.add_new_word(current_word, final_dictionary, ham_words, processed_words, spam_words, total_amount)
        # priorSpam = spam_amount / total_amount
        priorSpam = spam_count / len(files)
        self.logPrior = math.log(priorSpam / (1.0 - priorSpam))
        final_dictionary.sort(key=lambda x: x.p, reverse=True)
        self.dictionary = final_dictionary
        return self.dictionary, self.logPrior

    def add_new_word(self, current_word, final_dictionary, ham_words, processed_words, spam_words, total_amount):
        if not (current_word in processed_words):
            processed_words.append(current_word)
            ham_count_of_current_word = ham_words.count(current_word)
            spam_count_of_current_word = spam_words.count(current_word)
            occurrence = ham_count_of_current_word + spam_count_of_current_word

            if ham_count_of_current_word == 0:
                ham_count_of_current_word = 1
                occurrence += 1
            if spam_count_of_current_word == 0:
                spam_count_of_current_word = 1
                occurrence += 1
            new_word = wordCounter(current_word, ham_count_of_current_word, spam_count_of_current_word,
                                   occurrence / total_amount)
            final_dictionary.append(new_word)

    def classify(self, message: str, number_of_features: int) -> bool:
        '''
        :param message: Input email message as a string
        :param number_of_features: Number of features to be used from the trained dictionary
        :return: True if classified as SPAM and False if classified as HAM
        '''
        txt = np.array(self._extractWords(message))
        # TODO: Implement classification function
        P = self.logPrior
        features_dictionary = []
        for n in range(number_of_features):
            features_dictionary.append(self.dictionary[n])
            features_dictionary.append(self.dictionary[len(self.dictionary)-n-1])
        for i in range(len(txt)):
            for l in range(len(features_dictionary)):
                current_feature = features_dictionary[l]
                if txt[i] == current_feature.word:
                    P += np.log(current_feature.numOfSpamWords / current_feature.numOfHamWords)
        return P > 0

    def classifyAndEvaluateAllInFolder(self, msgDirectory: str, number_of_features: int,
                                       fileFormat: str = '*.txt') -> float:
        '''
        :param msgDirectory: Directory to email files that should be classified
        :param number_of_features: Number of features to be used from the trained dictionary
        :return: Classification accuracy
        '''
        files = sorted(glob.glob(msgDirectory + fileFormat))
        corr = 0  # Number of correctly classified messages
        ncorr = 0  # Number of falsely classified messages
        # TODO: Classify each email found in the given directory and figure out if they are correctly or falsely classified
        # TODO: Hint - look at the filenames to figure out the ground truth label
        for file in files:
            email = open(file, 'r').read()
            if self.classify(email, number_of_features):
                if "spm" in file:
                    corr += 1
                else:
                    ncorr += 1
            else:
                if "spm" in file:
                    ncorr += 1
                else:
                    corr += 1
        return corr / (corr + ncorr)

    def printMostPopularSpamWords(self, num: int) -> None:
        print("{} most popular SPAM words:".format(num))
        # TODO: print the 'num' most used SPAM words from the dictionary
        dict = sorted(self.dictionary, key=lambda x: x.numOfSpamWords, reverse=True)
        for i in range(num):
            print("Wort:" + dict[i].word + "\t Anzahl:" + str(dict[i].numOfSpamWords))

    def printMostPopularHamWords(self, num: int) -> None:
        print("{} most popular HAM words:".format(num))
        dict = sorted(self.dictionary, key=lambda x: x.numOfHamWords, reverse=True)
        for i in range(num):
            print("Wort:" + dict[i].word + "\t Anzahl:" + str(dict[i].numOfHamWords))

    def printMostindicativeSpamWords(self, num: int) -> None:
        print("{} most distinct SPAM words:".format(num))
        # TODO: print the 'num' most indicative SPAM words from the dictionary
        dict = sorted(self.dictionary, key=lambda x: x.p, reverse=True)
        for i in range(num):
            print("Wort:" + dict[i].word + "\t Anzahl:" + str(dict[i].numOfSpamWords) +  "\t P:" + str(dict[i].p))

    def printMostindicativeHamWords(self, num: int) -> None:
        print("{} most distinct HAM words:".format(num))
        # TODO: print the 'num' most indicative HAM words from the dictionary
        dict = sorted(self.dictionary, key=lambda x: x.p)
        for i in range(num):
            print("Wort:" + dict[i].word + "\t Anzahl:" + str(dict[i].numOfHamWords) + "\t P:" + str(dict[i].p))
