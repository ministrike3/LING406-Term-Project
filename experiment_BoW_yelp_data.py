import nltk
import nltk.sentiment
import nltk.corpus
import math
import sklearn
import sklearn.tree
import glob
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


def statistically_split_word_lists(neg, pos, max_length, thresh):
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
    max_length=min(len(neg_words),len(pos_words))
    neg_list = sorted(neg_words, key=neg_words.get, reverse=True)[:max_length]
    pos_list = sorted(pos_words, key=pos_words.get, reverse=True)[:max_length]
    new_neg={k:neg_words[k] for k in neg_list}
    new_pos = {k: pos_words[k] for k in pos_list}
    return new_neg,new_pos


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
    # print(negative_chance, positive_chance)
    if negative_chance > positive_chance:
        return 0
    else:
        return 1

def create_sklearn_data_partitions():
    neg_reviews = sorted(glob.glob('./review_polarity/txt_sentoken/neg/*'))
    pos_reviews = sorted(glob.glob('./review_polarity/txt_sentoken/pos/*'))
    return neg_reviews[:750]+pos_reviews[:750],neg_reviews[750:]+pos_reviews[750:]

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
    negative_movie_review_files, positive_movie_review_files = get_movie_review_data()
    name = './word_bags/nothing_neg_word_bag.txt'
    #negative_bag_of_words = create_bag_of_words(name, negative_movie_review_files[:750])
    #print(len(negative_bag_of_words.keys()))
    negative_bag_of_words = load_bag_of_words(name)
    print(len(negative_bag_of_words.keys()))

    name = './word_bags/nothing_pos_word_bag.txt'
    #positive_bag_of_words = create_bag_of_words(name, positive_movie_review_files[:750])
    #print(len(positive_bag_of_words.keys()))
    positive_bag_of_words = load_bag_of_words(name)
    print(len(positive_bag_of_words.keys()))

    neg_keys, pos_keys = statistically_split_word_lists(negative_bag_of_words, positive_bag_of_words, max_length, threshold_for_selection)
    print(len(neg_keys))
    print(len(pos_keys))

    ###### BOW CLASSIFIER
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

    negative_score = 0
    positive_score = 0
    for file in negative_movie_review_files[750:]:
        negative_score += run_bag_of_words_classification(neg_keys, pos_keys, negative_bag_of_words, positive_bag_of_words, max_neg, max_pos, weighting, file)
    print('Negative Score: '+ str((250-negative_score)/250))
    for file in positive_movie_review_files[750:]:
        positive_score += run_bag_of_words_classification(neg_keys, pos_keys, negative_bag_of_words, positive_bag_of_words, max_neg, max_pos,weighting, file)
    print('Positive Score: ' + str(positive_score/250))

    ##### NAIVE BAYES CLASSIFIER
    print('Naive Bayes')
    negative_score = 0
    positive_score = 0
    neg_size = sum(neg_keys.values()) + len(neg_keys.keys())
    pos_size = sum(pos_keys.values()) + len(pos_keys.keys())
    for file in negative_movie_review_files[750:]:
        negative_score += naive_bayes(neg_keys, pos_keys, neg_size, pos_size, file)
    print('Negative Score: '+ str((250-negative_score)/250))
    for file in positive_movie_review_files[750:]:
        positive_score += naive_bayes(neg_keys, pos_keys, neg_size, pos_size, file)
    print('Positive Score: ' + str(positive_score/250))

    #### SVM CLassifier
    print('SVM')
    vocab = generate_svm_featureset(neg_keys,pos_keys)
    train, test = create_sklearn_data_partitions()
    count_vect = sklearn.feature_extraction.text.CountVectorizer(input='filename',tokenizer=blah_tokenize,vocabulary=vocab)
    clf = sklearn.linear_model.SGDClassifier()

    X_train_counts = count_vect.fit_transform(train)
    y_cats = [0]*750+[1]*750
    clf.fit(X_train_counts,y_cats)

    X_test_counts = count_vect.fit_transform(test)
    scores = clf.predict(X_test_counts)
    negative_score = 0
    positive_score = 0
    for i in range(0, 250):
        negative_score += scores[i]
    for i in range(250, 500):
        positive_score += scores[i]
    print('Negative Score: ' + str((250 - negative_score) / 250))
    print('Positive Score: ' + str(positive_score/250))

    #### Logistic Regression CLassifier
    print('Logistic Regression Classifier')
    clf = sklearn.linear_model.SGDClassifier()
    y_cats = [0]*750+[1]*750
    clf.fit(X_train_counts,y_cats)
    scores = clf.predict(X_test_counts)
    neg_score = 0
    pos_score = 0
    negative_score = 0
    positive_score = 0
    for i in range(0, 250):
        negative_score += scores[i]
    for i in range(250, 500):
        positive_score += scores[i]
    print('Negative Score: ' + str((250 - negative_score) / 250))
    print('Positive Score: ' + str(positive_score/250))

    #### Decision Tree Classifier
    print('Decision Tree Classifier')
    clf = sklearn.tree.DecisionTreeClassifier()
    y_cats = [0]*750+[1]*750
    clf.fit(X_train_counts,y_cats)
    scores = clf.predict(X_test_counts)
    neg_score = 0
    pos_score = 0
    negative_score = 0
    positive_score = 0
    for i in range(0, 250):
        negative_score += scores[i]
    for i in range(250, 500):
        positive_score += scores[i]
    print('Negative Score: ' + str((250 - negative_score) / 250))
    print('Positive Score: ' + str(positive_score/250))