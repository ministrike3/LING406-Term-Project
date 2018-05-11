import nltk
import nltk.sentiment
import nltk.corpus
import math
import sklearn
import sklearn.tree
import glob
import string
from experiment_BoW import statistically_split_word_lists
import nltk.tokenize.moses

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


def generate_svm_featureset(neg_bow,pos_bow):
    review_word_index = []
    for word in neg_bow.keys():
        review_word_index.append(word)
    for word in pos_bow.keys():
        review_word_index.append(word)
    review_word_index = sorted(list(set(review_word_index)))
    return review_word_index


def blah_tokenize(line):
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
    return(line)

if __name__ == "__main__":
    punc = int(input("should punctuation be removed"))
    signif = int(input("should insignficant words be removed"))
    alpha = int(input("should only alpha words be considered"))
    stopword = int(input("should stopwords be removed"))
    negation = int(input("should we mark negations"))
    weighting = int(input("should we use weighting"))
    max_length = int(input("define a max_length/vocab size for the BoW"))
    threshold_for_selection = float(input("Define a cutoff max ie 1.2"))



    reviews=get_yelp_data()
    sorted_reviews=assign_yelp_data(reviews)
    sorted_reviews=move_half_star_reviews_up(sorted_reviews)

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
    red=((900-x)/900)
    x=0
    for review in pos_reviews_test:
        x+=run_bag_of_words_classification(neg_bag,pos_bag,max_neg,max_pos,review,weighting)
    blue=(x/1200)
    print((red+blue)/2)

    print('Naive Bayes')

    neg_size = sum(neg_keys.values()) + len(neg_keys.keys())
    pos_size = sum(pos_keys.values()) + len(pos_keys.keys())

    x=0
    for review in neg_reviews_test:
        x+=naive_bayes(neg_keys, pos_keys, neg_size, pos_size, review)
    blue=((900-x)/900)
    x=0
    for review in pos_reviews_test:
        x+=naive_bayes(neg_keys, pos_keys, neg_size, pos_size, review)
    red=(x/1200)
    print((red+blue)/2)

    #### SVM CLassifier
    print('SVM')
    vocab = generate_svm_featureset(neg_keys, pos_keys)

    train=[]
    for index in range(0,len(neg_reviews_train)):
        train.append(" ".join(neg_reviews_train[index]))
    for index in range(0,len(pos_reviews_train)):
        train.append(" ".join(pos_reviews_train[index]))
    test=[]
    for index in range(0,len(neg_reviews_test)):
        test.append(" ".join(neg_reviews_test[index]))
    for index in range(0,len(pos_reviews_test)):
        test.append(" ".join(pos_reviews_test[index]))
    count_vect = sklearn.feature_extraction.text.CountVectorizer(input='content', tokenizer=blah_tokenize,
                                                                 vocabulary=vocab)
    clf = sklearn.linear_model.SGDClassifier()

    X_train_counts = count_vect.fit_transform(train)
    y_cats = [0] * len(neg_reviews_train) + [1] * len(pos_reviews_train)
    clf.fit(X_train_counts, y_cats)

    X_test_counts = count_vect.fit_transform(test)
    scores = clf.predict(X_test_counts)
    negative_score = 0
    positive_score = 0
    for i in range(0, len(neg_reviews_test)):
        negative_score += scores[i]
    for i in range(len(neg_reviews_test), len(neg_reviews_test)+len(pos_reviews_test)):
        positive_score += scores[i]
    blue=((len(neg_reviews_test) - negative_score) / len(neg_reviews_test))
    red=((positive_score / len(pos_reviews_test)))

    print((blue+red)/2)

    #### Logistic
    print('Logistic Regression')
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(X_train_counts, y_cats)
    scores = clf.predict(X_test_counts)
    negative_score = 0
    positive_score = 0
    for i in range(0, len(neg_reviews_test)):
        negative_score += scores[i]
    for i in range(len(neg_reviews_test), len(neg_reviews_test) + len(pos_reviews_test)):
        positive_score += scores[i]
    blue = ((len(neg_reviews_test) - negative_score) / len(neg_reviews_test))
    red = (positive_score / len(pos_reviews_test))
    print((blue + red) / 2)

    #### Tree
    print('Decision Tree')
    clf = sklearn.tree.DecisionTreeClassifier()
    clf.fit(X_train_counts, y_cats)
    scores = clf.predict(X_test_counts)
    negative_score = 0
    positive_score = 0
    for i in range(0, len(neg_reviews_test)):
        negative_score += scores[i]
    for i in range(len(neg_reviews_test), len(neg_reviews_test) + len(pos_reviews_test)):
        positive_score += scores[i]
    blue = ((len(neg_reviews_test) - negative_score) / len(neg_reviews_test))
    red = ((positive_score / len(pos_reviews_test)))
    print((blue + red) / 2)