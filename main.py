#Goal : Create Word Vectors from a game of thrones #dataset of text files

#Step 1 - Import Libraries
from __future__ import absolute_import , division , print_function
# it is to call functions from python 2 and python 3 , to use both at the same time 

import codecs
#for word encoding

import glob
#used fore regex , used for searching texts in huge databases and text files

import multiprocessing

import os 
#for reading and writing files

import pprint
#to make the file human readable

import re
#regular expression

import nltk 
#natural language toolkit

import gensim.models.word2vec as w2v 
#word to Vectors

import sklearn.manifold 
#dimensonality reduction

import numpy as np 
#matrix math

import matplotlib.pyplot as plt 
#plotting

import pandas as pd 

import seaborn as sns 
#visiualizatin of datasets


#Step 2 Process data

#clean data 
nltk.download('punkt') #download tokenizer
nltk.download('stopwords') #downloading stopwords


#load books from files
book_filenames = sorted(glob.glob("data/clean/*.txt"))
print("Found Books")
book_filenames

#Combine the books into one String 
corpus_raw = u""
for book_filename in book_filenames:
  print("Reading {0}...".format(book_filename))
  with codecs.open(book_filename , "r" , "utf-8") as book_file:
    corpus_raw += book_file.read()
  print("Corpus is now {0} characters long".format(len(corpus_raw)))
  print()
  
#above we created a corpus of all the text files by converting them to utf8 format 

#split the corpus into sentences
# use tokenizer taken from nltk

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_sentences = tokenizer.tokenize(corpus_raw)

#remove unncessary punctuations , split into words
def sentences_to_wordlist(raw):
  clean = re.sub("[^a-zA-Z]"," ", raw)
  words = clean.split()
  return words
  
#sentence where each word is tokenized 
sentences = []
for raw_sentence in raw_sentences:
  if len(raw_sentences) > 0 :
    sentences.append(sentences_to_wordlist(raw_sentence))
    
print(raw_sentences[5])
print(sentences_to_wordlist(raw_sentences[5]))

token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains (0:,} tokens".format(token_count))


#train word2vec

#hyperparameters
#more dimensions , more computationally expensive although its more accurate meaning more dimenstinos = more generalized
#dimensionality of the resulting word Vectors
num_features = 300

#minimize word count threshold for each word embedding
min_word_count = 3

#Number of threads to run in parallel
num_workers = multiprocessing.cpu_count()

#context window length is the window size of how many words should be displayed 
context_size = 7

#downsample setting for frequent words
downsampling - le-3

#seed to generate random numbers
#deterministic good for debugging
seed  = 1


thrones2vec = w2v.Word2Vec(sg = 1, seed = seed , workers = num_workers , size = num_features  , min_count = min_word_count , window = context_size , sample = downsampling)

thrones2vec.build_vocab(sentences)

print("Word2Vec vocabulary length:" ,len(thrones2vec.vocab))


#train the model
thrones2vec.train(sentences)

#save file 
if not os.path.exists("Trained"):
  os.makedirs("Trained")
  
thrones2vec.save(os.path.join("Trained","thrones2vec.w2v"))

#exploring the trained model
thrones2vec = w2v.Word2Vec.load(os.path.join("Trained", "thrones2vec.w2v"))

#compress the word vectors into 2D space and plot them

tane = sklearn.mainfold.TSNE(n_components = 2 , random_state = 0)
#put all of it in a matrix
all_world_vectors_matrix = thrones2vec.syn0

#train t-SNE ( T - Stochastic Neighboring Embedding)

all_world_vectors_matrix_2d = tsne.fit_transform(all_world_vectors_matrix)

#plot the big picture

points = pd.DataFrame([
  (word , coords[0] , coords[1])
  for word , coords in [
    (word , all_world_vectors_matrix_2d[thrones2vec[word].index])
    for word in thrones2vec.vocab]
  ],
  columns = ["word" , "x" , "y"])

points.head(10)

sns.set_context("poster")
points.plot.scatter("x","y",s = 10 , figsize = (20,12))

#zoom into some interesting places
def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)
        
        
plot_region(x_bounds=(4.0, 4.2), y_bounds=(-0.5, -0.1))

#food products grouping
plot_region(x_bounds=(0, 1), y_bounds=(4, 4.5))

#semantic similarities between book characters
thrones2vec.most_similar("Stark")

thrones2vec.most_similar("Aerys")

#linear relationship between word pairs

def nearest_similarity_cosmul(start1, end1, end2):
    similarities = thrones2vec.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2
    
nearest_similarity_cosmul("Stark", "Winterfell", "Riverrun")
nearest_similarity_cosmul("Jaime", "sword", "wine")
nearest_similarity_cosmul("Arya", "Nymeria", "dragons")