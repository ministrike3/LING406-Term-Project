import glob
# import pandas as pd
import nltk
import nltk.sentiment
import string


def get_movie_review_data():
    neg_reviews = sorted(glob.glob('./review_polarity/txt_sentoken/neg/*'))
    pos_reviews = sorted(glob.glob('./review_polarity/txt_sentoken/pos/*'))

    return neg_reviews, pos_reviews


def create_bag_of_words(_name, reviews):
    word_bag = {}
    for review in reviews:
        with open(review, 'r') as f:
            line = f.read()
            line = line.split(" ")
            if punc:
                line = [word.translate(str.maketrans('', '', string.punctuation)) for word in line]
            if alpha:
                line = [word for word in line if word.isalpha()]
            if signif:
                line = [word for word in line if len(word) > 1]
            if stopword:
                line = [word for word in line if word not in set(nltk.corpus.stopwords.words('english'))]
            if negation:
                line = nltk.sentiment.util.mark_negation(line)
            for word in line:
                word_bag[word] = word_bag.get(word, 0) + 1
    with open(_name, 'w') as f:
        for _ in sorted(word_bag, key=word_bag.get, reverse=True):
            f.write("%s %d\n" % (_, word_bag[_]))
    return word_bag


def load_bag_of_words(_file):
    word_bag = {}
    with open(_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, count = line.split(" ")
            word_bag[key] = int(count)

    return word_bag


def statistically_split_word_lists(neg, pos, thresh):
    pos_words = {}
    neg_words = {}
    for key in neg.keys():
        if key in pos.keys():
            if int(pos[key]) >= thresh*int(neg[key]):
                pos_words[key] = pos[key]
            elif int(neg[key]) > thresh*int(pos[key]):
                neg_words[key] = neg[key]
        else:
            neg_words[key] = neg[key]
    for key in pos.keys():
        if key not in neg.keys():
            pos_words[key] = pos[key]
    if thresh == 1:
        return neg, pos
    else:
        return neg_words, pos_words


def run_bag_of_words_classification(neg, pos, negscore, posscore, max_neg_score, max_pos_score, _weighting, _file):
    score = 0
    with open(_file, 'r') as f:
        line = f.read()
        line = line.split(" ")
        if punc:
            line = [word.translate(str.maketrans('', '', string.punctuation)) for word in line]
        if alpha:
            line = [word for word in line if word.isalpha()]
        if signif:
            line = [word for word in line if len(word) > 1]
        if negation:
            line = nltk.sentiment.util.mark_negation(line)
        if stopword:
            line = [word for word in line if word not in set(nltk.corpus.stopwords.words('english'))]

        for word in line:
            score_incrementer = 1
            if word in neg:
                if _weighting:
                    score_incrementer = int(negscore[word])/max_neg_score
                score -= score_incrementer
            if word in pos:
                if _weighting:
                    score_incrementer = int(posscore[word]) / max_pos_score
                score += score_incrementer
    if score >= 0:
        return 1
    else:
        return 0

if __name__ == "__main__":
    punc = int(input("should punctuation be removed"))
    signif = int(input("should insignficant words be removed"))
    alpha = int(input("should only alpha words be considered"))
    stopword = int(input("should stopwords be removed"))
    negation = int(input("should we mark negations"))
    weighting = int(input("should we use weighting"))
    max_length = int(input("define a max_length/vocab size for the BoW"))
    threshold_for_selection = float(input("Define a cutoff max ie 1.2"))
    negative_movie_review_files, positive_movie_review_files = get_movie_review_data()
    name = './word_bags/baseline_neg_word_bag.txt'
    # negative_bag_of_words = create_bag_of_words(name, negative_movie_review_files[:750])
    # print(len(negative_bag_of_words.keys()))
    negative_bag_of_words = load_bag_of_words(name)
    # print(len(negative_bag_of_words.keys()))

    name = './word_bags/baseline_pos_word_bag.txt'
    # positive_bag_of_words = create_bag_of_words(name, positive_movie_review_files[:750])
    # print(len(positive_bag_of_words.keys()))
    positive_bag_of_words = load_bag_of_words(name)
    # print(len(positive_bag_of_words.keys()))

    neg_keys, pos_keys = statistically_split_word_lists(negative_bag_of_words, positive_bag_of_words, threshold_for_selection)
    # print(len(neg_keys))
    # print(len(pos_keys))
    # max_length=max(len(neg_keys),len(pos_keys))
    neg_keys = sorted(neg_keys, key=neg_keys.get, reverse=True)[:max_length]
    max_neg = 0
    for i in neg_keys:
        max_neg += int(negative_bag_of_words[i])
    max_neg /= int(negative_bag_of_words[neg_keys[0]])

    pos_keys = sorted(pos_keys, key=pos_keys.get, reverse=True)[:max_length]
    max_pos = 0
    for i in pos_keys:
        max_pos += int(positive_bag_of_words[i])
    max_pos /= int(positive_bag_of_words[pos_keys[0]])
    negative_score = 0
    positive_score = 0
    for file in negative_movie_review_files[750:]:
        negative_score += run_bag_of_words_classification(neg_keys, pos_keys, negative_bag_of_words, positive_bag_of_words, max_neg, max_pos, weighting, file)
    print((250-negative_score)/250)
    for file in positive_movie_review_files[750:]:
        positive_score += run_bag_of_words_classification(neg_keys, pos_keys, negative_bag_of_words, positive_bag_of_words, max_neg, max_pos,weighting, file)
    print(positive_score/250)
    print((((250-negative_score)/250)+(positive_score/250))/2)
