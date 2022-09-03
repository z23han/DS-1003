#!/usr/bin/env python3


import util, load, pickle
import numpy as np
import matplotlib.pyplot as plt


def get_all_words(reviews):
    words = set()
    for sentence in reviews:
        words |= set(sentence[:-1])
    return words


def build_features(words):
    words_list = list(words)
    return dict([(words_list[i], i) for i in range(len(words))])


def fill_data(reviews, X_data, y_data, feature_map):
    for i in range(len(reviews)):
        words = reviews[i][:-1]
        label = reviews[i][-1]

        bag_map = util.convert(words)
        for k, v in bag_map.items():
            X_data[i][feature_map[k]] = v
        y_data[i] = label
    print("finished fill_data")



def prepare_data():
    train_data, val_data = None, None
    with open(load.train_file_name, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(load.val_file_name, 'rb') as f:
        val_data = pickle.load(f)
    
    words = get_all_words(train_data) | get_all_words(val_data)
    feature_map = build_features(words)

    X_train, y_train = np.zeros((len(train_data), len(words))), np.zeros(len(train_data))
    X_test, y_test = np.zeros((len(val_data), len(words))), np.zeros(len(val_data))

    fill_data(train_data, X_train, y_train, feature_map)
    fill_data(val_data, X_test, y_test, feature_map)

    return X_train, y_train, X_test, y_test


def test_pegasos(X_train, y_train, X_test, y_test, lambd, num_iter):
    print("lambda={}, num_iter={}".format(lambd, num_iter))

    weights = util.pegasos(X_train, y_train, lambd=lambd, num_iter=num_iter)
    y_pred = np.dot(X_test, weights)
    y_pred_hat = np.array([1 if y0 >= 0 else -1 for y0 in y_pred])

    accuracy = float((y_test == y_pred_hat).sum()) / len(y_test)
    print("accuracy={}".format(accuracy))

    return accuracy




def main():
    num_iter = 500
    lambd_list = np.arange(-2, 2, 0.25, dtype=float)
    X_train, y_train, X_test, y_test = prepare_data()

    accuracy_list = []
    for lambd in lambd_list:
        accuracy_list.append(test_pegasos(X_train, y_train, X_test, y_test, lambd, num_iter))
    
    plt.plot(lambd_list, accuracy_list)
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.show()


if __name__ == '__main__':
    main()

