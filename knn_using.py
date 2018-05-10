import nltk.sentiment
import nltk.corpus
import math
import sklearn
import pandas
import numpy
import glob
import string
from experiment_BoW import *

def get_movie_review_data():
    neg_reviews = sorted(glob.glob('./review_polarity/txt_sentoken/neg/*'))
    pos_reviews = sorted(glob.glob('./review_polarity/txt_sentoken/pos/*'))

    return neg_reviews[:750]+pos_reviews[:750],neg_reviews[750:]+pos_reviews[750:]

def create_svm_dataset():
    _name = './word_bags/baseline_neg_word_bag.txt'
    _negative_bag_of_words = load_bag_of_words(_name)

    _name = './word_bags/baseline_pos_word_bag.txt'
    _positive_bag_of_words = load_bag_of_words(_name)

    review_word_index = []
    for word in _negative_bag_of_words.keys():
        review_word_index.append(word)
    for word in _positive_bag_of_words.keys():
        review_word_index.append(word)
    review_word_index = sorted(list(set(review_word_index)))
    return review_word_index


def blah_tokenize(line):
    line = line.split(" ")
    line = [word.translate(str.maketrans('', '', string.punctuation)) for word in line]
    line = [word for word in line if word.isalpha()]
    line = [word for word in line if len(word) > 1]
    line = [word for word in line if word not in set(nltk.corpus.stopwords.words('english'))]
    return(line)


if __name__ == "__main__":
    vocab = create_svm_dataset()
    negative_movie_review_files, positive_movie_review_files = get_movie_review_data()
    count_vect = sklearn.feature_extraction.text.CountVectorizer(input='filename',tokenizer=blah_tokenize,vocabulary=vocab)
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3)

    X_train_counts = count_vect.fit_transform(negative_movie_review_files)
    y_cats = [0]*750+[1]*750
    clf.fit(X_train_counts,y_cats)

    X_test_counts = count_vect.fit_transform(positive_movie_review_files)
    scores = clf.predict(X_test_counts)
    neg_score = 0
    pos_score = 0
    for i in range(0, 250):
        neg_score += scores[i]
    for i in range(250, 500):
        pos_score += scores[i]
    print((250-neg_score)/250)
    print((pos_score)/250)