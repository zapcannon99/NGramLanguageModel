# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:21:56 2019

@author: Tenkichi
"""

import re
import nltk
nltk.download('punkt')
import math
import numpy as np
from os import path as ospath
import copy
import random
random.seed(1)

### GLOBAL DEFINITIONS -------------------------------------------------------

stok = "<s>" #start token
etok = "</s>" #end token
unktok = "<unk>"

def get_ngrams(n, text):
    #tokens = nltk.word_tokenize(text)
    temp = copy.deepcopy(text)
    for i in range(1,n):
        temp.insert(0, stok)
    temp.append(etok)
        
    for i in range(n-1, len(temp)-1):
        yield (temp[i], tuple(temp[i-n+1:i]))
        
def text_prob(model, text, delta=0, masked=False):
    return model.text_prob(text, delta, masked)

def mask_rare(corpus):
    return NGramLM.mask_rare(corpus)

def perplexity(model, corpus_path, delta=0):
    file = open(corpus_path, encoding="utf-8-sig", mode="r")
    test_lines = file.readlines()
    N = 0
    terms = list()
    for line in test_lines:
        if(line.strip() != ""):
            line_list = line.split()
            N = N + len(line_list)  + 1 #Not counting start tokens as they are never considered as words, but the add 1 for line stop
            p = model.text_prob(line_list, delta)
            terms.append(p)
    neg_I = -np.sum(terms)/N
    return math.pow(2, neg_I)
            
def random_text(model, max_length=10, delta=0):
    model.random_text(max_length, delta)

def likeliest_text(model, max_length=10, delta=0):
    model.likeliest_text(max_length, delta)

### CLASS DEFINITIONS --------------------------------------------------------

class NGramLM:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = dict()       #n-grams seen in the training data
        self.ngram_total = 0
        self.context_counts = dict()     #contexts seen in the training data
        self.context_total = 0
        self.vocabulary = dict()    #keeps track of words seen in the training data
        self.vocabulary_total = 0
        self.vocabulary_size = 0
        self.delta = 0
        
        #Add the start and end tokens
        self.vocabulary[stok] = n-1
        self.vocabulary[etok] = 1
        
    def update(self, text):
        for ngram in get_ngrams(self.n, text):
            if ngram in self.ngram_counts:
                self.ngram_counts[ngram] = self.ngram_counts[ngram] + 1
            else:
                self.ngram_counts[ngram] = 1                
            self.ngram_total = self.ngram_total + 1
                
            if ngram[1] in self.context_counts:
                self.context_counts[ngram[1]] = self.context_counts[ngram[1]] + 1
            else:
                self.context_counts[ngram[1]] = 1
            self.context_total = self.context_total + 1
                
            if ngram[0] in self.vocabulary:
                self.vocabulary[ngram[0]] = self.vocabulary[ngram[0]] + 1
            else:
                self.vocabulary[ngram[0]] = 1
                self.vocabulary_size = self.vocabulary_size + 1    
            self.vocabulary_total = self.vocabulary_total + 1
        
    def create_ngramlm(n, corpus_path, masked=False):
        model = NGramLM(n)
        if masked:
            if not ospath.exists("masked_"+corpus_path):
                masked_corpus = mask_rare(open(corpus_path, encoding='utf-8-sig', mode="r").read())
                open("masked_"+corpus_path, encoding="utf-8-sig", mode="w").write(masked_corpus)
                
            path = "masked_"+corpus_path
        else:
            path = corpus_path
#        print(path)
        model.learnlinebyline(path)
        
        return model
    
    def learnlinebyline(self, path):
        for linelist in NGramLM.linebyline(path):
            self.update(linelist)
            
    def linebyline(path):
        file = open(path, encoding='utf-8-sig', mode="r")
        lines = file.readlines()
        
        for line in lines:
            stripped = line.strip()
            if(stripped != ''): #if we have an empty string, skip
                yield stripped.split()
                
    def word_prob(self, word, context, delta=0, masked=False):
        # we need the previous markov chain probabilities starting with the first word            
        # We have to do each count separately, in case something does not exist in dictionaries
        # Find the ngram
        if self.n==1:
            context_count = self.vocabulary_total
        elif context in self.context_counts:
            if masked:
                if unktok in context:
                    context_count = self.vocabulary[unktok]
                else:
                    context_count = self.context_counts[context]
            else:
                context_count = self.context_counts[context]
        else:
            if masked:
                context_count = self.vocabulary[unktok]
            else:
                if delta == 0:
                    return 1/self.vocabulary_size
                else:
                    context_count = 0
                
        if (word, context) in self.ngram_counts:
            ngram_count = self.ngram_counts[(word, context)]
        else:
            if masked:
                ngram_count = self.vocabulary[unktok]
            else:
                ngram_count = 0
        
        if delta == 0:
            return ngram_count/context_count
        else:
#            print((ngram_count + delta)/(context_count + delta*self.context_total))
            return (ngram_count + delta)/(context_count + delta*self.vocabulary_size)
        
    def text_prob(self, text, delta=0, masked=False):
        n = self.n # Grab the models n to for ease of programming
        probs = list()
        for ngram in get_ngrams(n, text):
#            print(ngram)
            p = self.word_prob(ngram[0], ngram[1], delta, masked)
#            print("{} {} {}".format(ngram[0], ngram[1], p))
            probs.append(math.log(p))
        return np.sum(np.asarray(probs))
    
    #------------- Part 2 class functions ------------
    
    # really, corpus will just be the path because it would be easier to just use the source
    def mask_rare(corpus):
        words = dict()
        masked_corpus = copy.deepcopy(corpus)
        splitted = corpus.split()
        for word in splitted:
            if word in words:
                words[word] = words[word] + 1
            else:
                words[word] = 1
        for word in words:
            if words[word] == 1:
                if ")" in word:
                    word = word.replace(")" ,"\\)")
                if "(" in word:
                    word = word.replace("(", "\\(")
                if "*" in word:
                    word = word.replace("*", "\\*")
                    
                print(word)
                masked_corpus = re.sub(word+r"(?=\s)", unktok, masked_corpus, count=1)
                print("word {} replaced".format(word))
        print("masked corpus: " + masked_corpus)
        return masked_corpus
            
    #------------- Part 4 class functions
    
    def random_word(self, context, delta=0):
        sorted_keys = sorted(self.vocabulary.keys())
        
        r = random.random()
        cumulative_probability = 0;
        for key in sorted_keys:
            prob = self.word_prob(key, context, delta)
#            print(prob, r, cumulative_probability)
            if 0 <= r < cumulative_probability + prob:
                return key
            else:
                cumulative_probability = cumulative_probability + prob
                
    def random_text(self, max_length=10, delta=0):
        sentence = list()
        for i in range(1, self.n):
            sentence.append(stok);
        for i in range(0, max_length):
            context = tuple(sentence[-(self.n-1):])
            word = self.random_word(context, delta)
#            print("word", word, context)
            sentence.append(word)
            if(word == etok):
                break
        
        seperator = " "
        result = seperator.join(sentence)
        return result
    
    def likeliest_word(self, context, delta=0):
        count_max = 0
        ngram_max = tuple()
        
        for word in self.vocabulary:
            ngram = tuple([word, context])
            if ngram in self.ngram_counts:
                if self.ngram_counts[ngram] > count_max:
                    count_max = self.ngram_counts[ngram]
                    ngram_max = ngram
        
        if(ngram_max==tuple()):
            return unktok
        else:
            return ngram_max[0]

    def likeliest_text(self, max_length=10, delta=0):
        sentence = list()
        for i in range(1, self.n):
            sentence.append(stok);
        for i in range(0, max_length):
            context = tuple(sentence[-(self.n-1):])
            word = self.likeliest_word(context, delta)
#            print("word", word, context)
            sentence.append(word)
            if(word == etok):
                break
        
        seperator = " "
        result = seperator.join(sentence)
        return result
    
    

class NGramInterpolator():
    def __init__(self, n, lambdas):
        self.n = n
        self.lambdas = lambdas
        self.NGramLMs = list()
        for i in range(1,n+1):
            self.NGramLMs.append(NGramLM(i))
    
    def train(self, corpus_path):
        file = open(corpus_path, encoding='utf-8-sig', mode="r")
        corpus = file.readlines()
        self.update(corpus)
    
    def update(self, text):
        for model in self.NGramLMs:
            for line in text:
                model.update(line.split())
            
    def word_prob(self, word, context, delta=0):
        terms = list()
        for i in range(0, self.n):
            model = self.NGramLMs[i]
            l = self.lambdas[i]
            if i==0:
                p = model.word_prob(word, tuple(), delta=delta)
            elif i==1:
                p = model.word_prob(word, context[-1], delta=delta)
            else:
                p = model.word_prob(word, context[-i:], delta=delta)
            terms.append(l*p)
        return np.sum(np.array(terms))
        
    def text_prob(self, text, delta=0):
        sentence = copy.deepcopy(text)
        terms = list()
        for ngram in get_ngrams(self.n, sentence):
            terms.append(math.log(self.word_prob(ngram[0], ngram[1], delta)))
        return np.sum(terms)
        
    





###--------------- MAIN FUNCTION ------------------------------------------
        
    
model_warpeace = NGramLM.create_ngramlm(3, "warpeace.txt")
test_sentence1 = "God has given it to me, let him who touches it beware!".split()
test_sentence2 = "Where is the prince, my Dauphin?".split()

print("Log probability for test sentence 1 is {}.".format(text_prob(model_warpeace, test_sentence1)))
print("Log probability for test sentence 2 cannot be computed")

model_masked = NGramLM.create_ngramlm(3, "warpeace.txt", True)

print("Log probability for test sentence 1 with masking is {}.".format(text_prob(model_masked, test_sentence1, masked=True)))
print("Log probability for test sentence 2 with masking is {}.".format(text_prob(model_masked, test_sentence2, masked=True)))

deltas = np.arange(.1, 1.1, .1)
probs1 = list()
probs2 = list()
for d in deltas:
    probs1.append(text_prob(model_warpeace, test_sentence1, delta=d))
    probs2.append(text_prob(model_warpeace, test_sentence2, delta=d))
    
print("deltas used: {}".format(deltas))
print("Sentence 1 log probabilities with Laplace smoothing: {}".format(probs1))
print("Sentence 2 log probabilities with Laplace smoothing: {}".format(probs2))

model_interpolator = NGramInterpolator(3, [.33, .33, .33])
model_interpolator.train("warpeace.txt")
prob1 = model_interpolator.text_prob(test_sentence1)
prob2 = model_interpolator.text_prob(test_sentence2)

print("Sentence 1 log probability with linear interpolation: {}".format(prob1))
print("Sentence 2 log probability with linear interpolation: {}".format(prob2))


model_shakespeare = NGramLM.create_ngramlm(3, "shakespeare.txt")
perplexity_smoothed = perplexity(model_shakespeare, "sonnets.txt", delta=0.5)
#perplexity_none = perplexity(model_shakespeare, "sonnets.txt")

print("Perplexity of smoothed model is {}".format(perplexity_smoothed))
print("Perplexity of unsmoothed model is indeterminable.")


perplexity_shakespeare = perplexity(model_shakespeare, "sonnets.txt", delta=0.5)
perplexity_warpeace = perplexity(model_warpeace, "sonnets.txt", delta=0.5)

print("Shakespeare model perplexity against sonnets.txt is {}".format(perplexity_shakespeare))
print("Warpeace model perplexity against sonnets.txt is {}".format(perplexity_warpeace))

for i in range(0, 5):
    print(model_shakespeare.random_text(max_length=10, delta=0.5))


bigram = NGramLM.create_ngramlm(2, "shakespeare.txt")
trigram = NGramLM.create_ngramlm(3, "shakespeare.txt")
quadgram = NGramLM.create_ngramlm(4, "shakespeare.txt")
pentgram = NGramLM.create_ngramlm(5, "shakespeare.txt")

bisentence = bigram.likeliest_text(10)
trisentence = trigram.likeliest_text(10)
quadsentence = quadgram.likeliest_text(10)
pentsentence = pentgram.likeliest_text(10)

print("bisentence:", bisentence)
print("trisentence:", trisentence)
print("quadsentence:", quadsentence)
print("pentsentence:", pentsentence)
