import pandas as pd
def load_train_data(train_path):
    train=pd.read_csv(train_path)
    return train
def load_test_data(test_path):
    test=pd.read_csv(test_path)
    return test