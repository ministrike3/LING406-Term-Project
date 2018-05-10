import nltk
import nltk.sentiment
import nltk.corpus
import math
import sklearn
import sklearn.tree
import glob
import string

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
            #if stopword:
            #    line = [word for word in line if word not in set(nltk.corpus.stopwords.words('english'))]
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
    negative = sorted_reviews[0]+sorted_reviews[1]+sorted_reviews[2]
    positive = sorted_reviews[3] + sorted_reviews[4]
    return negative, positive

def yelp_word_bag_generator(reviews):
    word_bag={}
    for review in reviews:
        for word in review:
            word_bag[word] = word_bag.get(word, 0) + 1
    with open('./word_bags/yelp/negative_word_bag.txt', 'w') as f:
        for _ in sorted(word_bag, key=word_bag.get, reverse=True):
            f.write("%s %d\n" % (_, word_bag[_]))

if __name__ == "__main__":
    punc=1
    alpha=1
    signif=1
    stopword=1
    reviews=get_yelp_data()
    sorted_reviews=assign_yelp_data(reviews)
    print(len(sorted_reviews[2]),len(sorted_reviews[3]))
    sorted_reviews=move_half_star_reviews_up(sorted_reviews)
    print(len(sorted_reviews[2]),len(sorted_reviews[3]))
    sorted_reviews=break_up_review_data(sorted_reviews)
    neg_reviews, pos_reviews= split_into_neg_pos(sorted_reviews)
    yelp_word_bag_generator(neg_reviews)

