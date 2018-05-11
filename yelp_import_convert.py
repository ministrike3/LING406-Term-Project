import nltk
import nltk.sentiment
import nltk.corpus
import math
import sklearn
import sklearn.tree
import glob
import string
from experiment_BoW import statistically_split_word_lists

def get_yelp_data():
    with open('./yelp_reviews/all_reviews.txt', 'r') as f:
        text = f.read()
        reviews = text.split("]]]")
        reviews.pop(-1)
    return(reviews)

def assign_yelp_data(reviews):
    bloop=[[],[],[],[],[]]
    for review in reviews:
        _,body=review.split('{{{\n',1)
        index=int(body[0])-1
        _,body = review.split('[[[',1)
        bloop[index].append(body)
    return(bloop)

def break_up_review_data(info):
    for review_stars in info:
        for index in range(0,len(review_stars)):
            review=review_stars[index]
            review=review.lower()
            review = review.replace("\n"," ")
            review=review.split(" ")
            line=review
            if punc:
                line = [word.translate(str.maketrans('', '', string.punctuation)) for word in line]
            if alpha:
                line = [word for word in line if word.isalpha()]
            if signif:
                line = [word for word in line if len(word) > 1]
            if stopword:
                line = [word for word in line if word not in set(nltk.corpus.stopwords.words('english'))]
            # if negation:
            #    line = nltk.sentiment.util.mark_negation(line)
            review_stars[index]=line
    return info

def move_half_star_reviews_up(sorted_reviews):
    keep_track_of_index=[]
    for index in range(0, len(sorted_reviews[2])):
        move_up=0
        review_og = sorted_reviews[2][index]
        review = review_og.lower()
        review = review.replace("\n", " ")
        review = review.split(" ")
        for word in review:
            if word=='3.5':
                sorted_reviews[3].append(review_og)
                keep_track_of_index.append(index)
                break
    for index in reversed(keep_track_of_index):
        sorted_reviews[2].pop(index)
    return sorted_reviews

def split_into_neg_pos(sorted_reviews):
    negative_train = sorted_reviews[0][:-300]+sorted_reviews[1][:-300]+sorted_reviews[2][:-300]
    positive_train = sorted_reviews[3][:-600] + sorted_reviews[4][:-600]

    negative_test = sorted_reviews[0][-300:] + sorted_reviews[1][-300:]+sorted_reviews[2][-300:]
    positive_test = sorted_reviews[3][-600:] + sorted_reviews[4][-600:]
    return negative_train, positive_train, negative_test,positive_test

def run_bag_of_words_classification(neg_bow,pos_bow,max_neg_score,max_pos_score,review,weighting):
    score_incrementer=1
    score=0
    for word in review:
        score_incrementer = 1
        if word in neg_bow.keys():
            if weighting:
              score_incrementer = int(neg_bow[word])/max_neg_score
            score -= score_incrementer
        if word in pos_bow.keys():
            if weighting:
               score_incrementer = int(pos_bow[word]) / max_pos_score
            score += score_incrementer
    if score >= 0:
        return 1
    else:
        return 0

def naive_bayes(neg_bow, pos_bow,neg_size,pos_size, review):
    negative_chance = 0
    positive_chance = 0

    for word in review:
        if word in neg_bow.keys():
            negative_chance += math.log(int(neg_bow[word])/neg_size)
        else:
            negative_chance += math.log(1/neg_size)
        if word in pos_bow.keys():
            positive_chance += math.log(int(pos_bow[word])/pos_size)
        else:
            positive_chance += math.log(1/pos_size)
    # print(negative_chance, positive_chance)
    if negative_chance > positive_chance:
        return 0
    else:
        return 1

def yelp_word_bag_generator(reviews,name):
    word_bag={}
    for review in reviews:
        for word in review:
            word_bag[word] = word_bag.get(word, 0) + 1
    with open(name, 'w') as f:
        for _ in sorted(word_bag, key=word_bag.get, reverse=True):
            f.write("%s %d\n" % (_, word_bag[_]))
    return word_bag
if __name__ == "__main__":
    punc=1
    alpha=1
    signif=1
    stopword=1
    max_length=1
    negation=0
    weighting=0
    threshold_for_selection=1
    reviews=get_yelp_data()
    sorted_reviews=assign_yelp_data(reviews)
    print(len(sorted_reviews[2]),len(sorted_reviews[3]))
    sorted_reviews=move_half_star_reviews_up(sorted_reviews)
    print(len(sorted_reviews[2]),len(sorted_reviews[3]))
    sorted_reviews = break_up_review_data(sorted_reviews)
    neg_reviews_train, pos_reviews_train, neg_reviews_test, pos_reviews_test = split_into_neg_pos(sorted_reviews)
    neg_bag = yelp_word_bag_generator(neg_reviews_train,'./word_bags/yelp/negative_word_bag.txt')
    pos_bag = yelp_word_bag_generator(pos_reviews_train,'./word_bags/yelp/positive_word_bag.txt')
    neg_keys, pos_keys = statistically_split_word_lists(neg_bag, pos_bag, max_length, threshold_for_selection)

    print('Bag of Words')
    max_neg = 0
    blah = 0
    for i in neg_keys:
        test=int(neg_keys[i])
        if test>blah:
            blah=test
        max_neg += test
    max_neg /= blah
    max_pos = 0
    blah = 0
    for i in pos_keys:
        test=int(pos_keys[i])
        if test>blah:
            blah=test
        max_pos += test
    max_pos /= blah

    x=0
    for review in neg_reviews_test:
        x+=run_bag_of_words_classification(neg_bag,pos_bag,max_neg,max_pos,review,weighting)
    print((900-x)/900)
    x=0
    for review in pos_reviews_test:
        x+=run_bag_of_words_classification(neg_bag,pos_bag,max_neg,max_pos,review,weighting)
    print(x/1200)

    print('Naive Bayes')

    neg_size = sum(neg_keys.values()) + len(neg_keys.keys())
    pos_size = sum(pos_keys.values()) + len(pos_keys.keys())

    x=0
    for review in neg_reviews_test:
        x+=naive_bayes(neg_keys, pos_keys, neg_size, pos_size, review)
    print((900-x)/900)
    x=0
    for review in pos_reviews_test:
        x+=naive_bayes(neg_keys, pos_keys, neg_size, pos_size, review)
    print(x/1200)

