import pandas as pd
import config

from sklearn import model_selection


def create_folds(data):

    # create a new column called kfold and fill it with -1
    data["kfold"] = -1

    # next step is to randomize the rows of this data
    data = data.sample(frac=1).reset_index(drop=True)

    # fetch targets
    y = data.label.values

    # initiate the kfold class from model seleciton module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        data.loc[v_, "kfold"] = f

    return data


if __name__ == "__main__":

    # Training data is in a CSV file called train.csv
    df = pd.read_csv(config.RAW_DATA)

    # create folds
    df = create_folds(df)

    df.to_csv(config.TRAINING_FILE, index=False)
