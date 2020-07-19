# training driver file
import os
import argparse

import config
import dispatcher

import joblib
import pandas as pd
from sklearn import metrics


def run(fold: int, model: str) -> None:
    # read the training file with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # training data is every other folder than one given
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is this fold's data
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the labels and convert train into numpy array
    x_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values

    # let's do similarly for validation as well
    x_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    # initialize decision tree
    clf = dispatcher.MODELS[model]

    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)

    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    # save the model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"df_{fold}_{model}.bin"))


if __name__ == "__main__":
    run(fold=0, model="randomforest")
    run(fold=1, model="extratrees")

