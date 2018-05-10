import nltk.sentiment
import nltk.corpus
from experiment_BoW import *
import math


def naive_bayes(neg_bow, pos_bow,neg_size,pos_size, _file):
    negative_chance = 0
    positive_chance = 0
    with open(_file, 'r') as f:
        line = f.read()
        line = line.split(" ")
        line = [word.translate(str.maketrans('', '', string.punctuation)) for word in line]
        line = [word for word in line if word.isalpha()]
        line = [word for word in line if len(word) > 1]
        line = [word for word in line if word not in set(nltk.corpus.stopwords.words('english'))]
        # if negation:
        #    line = nltk.sentiment.util.mark_negation(line)

        for word in line:
            if word in neg_bow.keys():
                negative_chance += math.log(int(neg_bow[word])/neg_size)
            else:
                negative_chance += math.log(1/neg_size)
            if word in pos_bow.keys():
                positive_chance += math.log(int(pos_bow[word])/pos_size)
            else:
                positive_chance += math.log(1/pos_size)
    #print(negative_chance, positive_chance)
    if negative_chance > positive_chance:
        return 0
    else:
        return 1



if __name__ == "__main__":

    negative_movie_review_files, positive_movie_review_files = get_movie_review_data()
    name = './word_bags/baseline_neg_word_bag.txt'
    negative_bag_of_words = load_bag_of_words(name)

    name = './word_bags/baseline_pos_word_bag.txt'
    positive_bag_of_words = load_bag_of_words(name)

    negative_score = 0
    positive_score = 0
    neg_size = sum(negative_bag_of_words.values())+len(negative_bag_of_words.keys())
    pos_size = sum(positive_bag_of_words.values())+len(positive_bag_of_words.keys())
    for file in negative_movie_review_files[750:]:
        negative_score += naive_bayes(negative_bag_of_words, positive_bag_of_words, neg_size, pos_size, file)
    print(negative_score)
    for file in positive_movie_review_files[750:]:
        positive_score += naive_bayes(negative_bag_of_words, positive_bag_of_words, neg_size, pos_size, file)
    print(positive_score)
    print((((250-negative_score)/250)+(positive_score/250))/2)
