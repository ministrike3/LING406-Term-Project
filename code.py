import glob
# import pandas as pd
import nltk
import string


def get_movie_review_data():
    neg_reviews = sorted(glob.glob('./review_polarity/txt_sentoken/neg/*'))
    pos_reviews = sorted(glob.glob('./review_polarity/txt_sentoken/pos/*'))

    return neg_reviews, pos_reviews


def create_bag_of_words(name, reviews):
    word_bag = {}
    for review in reviews:
        with open(review, 'r') as f:
            line = f.read()
            line = line.split(" ")
            line = [word.translate(str.maketrans('', '', string.punctuation)) for word in line]
            line = [word for word in line if word.isalpha()]
            line = [word for word in line if len(word) > 1]
            line = [word for word in line if word not in set(nltk.corpus.stopwords.words('english'))]
            for word in line:
                word_bag[word] = word_bag.get(word, 0) + 1
    #with open(name,'w') as f:
    #    for i in sorted(word_bag, key=word_bag.get,reverse=True):
    #        f.write("%s %d\n" %(i,word_bag[i]))
    return word_bag

def load_bag_of_words(file):
    word_bag = {}
    with open(file, 'r') as f:
        lines=f.readlines()
        for line in lines:
            key, count = line.split(" ")
            word_bag[key]=count

    return word_bag

def statistically_split_word_lists(neg, pos):
    pos_words = {}
    neg_words = {}
    for key in neg.keys():
        if key in pos.keys():
            if pos[key] >= neg[key]:
                pos_words[key]=pos[key]
            else:
                neg_words[key]=neg[key]
        else:
            neg_words[key] = neg[key]
    for key in pos.keys():
        if key not in neg.keys():
            pos_words[key] = pos[key]
    return neg_words, pos_words


def run_bag_of_words_classification(neg, pos, _file):
    score = 0
    with open(_file, 'r') as f:
        line = f.read()
        line = line.split(" ")
        line = [word.translate(str.maketrans('', '', string.punctuation)) for word in line]
        line = [word for word in line if word.isalpha()]
        line = [word for word in line if len(word) > 1]
        for word in line:
            if word in neg:
                score -= 1
            if word in pos:
                score += 1
    if score >= 0:
        return 1
    else:
        return 0

if __name__ == "__main__":
    negative_movie_review_files, positive_movie_review_files = get_movie_review_data()
    name= './neg_word_bag.txt'
    #negative_bag_of_words = create_bag_of_words(name, negative_movie_review_files[:900])
    #print(len(negative_bag_of_words.keys()))
    negative_bag_of_words=load_bag_of_words(name)
    print(len(negative_bag_of_words.keys()))

    name= './pos_word_bag.txt'
    #positive_bag_of_words = create_bag_of_words(name, positive_movie_review_files[:900])
    #print(len(positive_bag_of_words.keys()))
    positive_bag_of_words=load_bag_of_words(name)
    print(len(positive_bag_of_words.keys()))

    neg_keys, pos_keys = statistically_split_word_lists(negative_bag_of_words, positive_bag_of_words)
    print(len(neg_keys))
    print(len(pos_keys))
    neg_keys = sorted(neg_keys, key=neg_keys.get, reverse=True)[:len(neg_keys)]
    pos_keys = sorted(pos_keys, key=pos_keys.get, reverse=True)[:len(neg_keys)]
    negative_score = 0
    positive_score = 0
    for file in negative_movie_review_files[900:]:
        negative_score += run_bag_of_words_classification(neg_keys, pos_keys, file)
    print((100-negative_score)/100)
    for file in positive_movie_review_files[900:]:
        positive_score += run_bag_of_words_classification(neg_keys, pos_keys, file)
    print(positive_score/100)
