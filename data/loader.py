import pandas as pd
from sklearn.model_selection import train_test_split


def load_train_val(csv_path, test_size=0.2, random_state=42):
    raw_data = pd.read_csv(csv_path)
    labels = raw_data.iloc[:, 0].values
    images = raw_data.iloc[:, 1:].values
    return train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )


def load_test(csv_path):
    raw_data = pd.read_csv(csv_path)
    return raw_data.values


if __name__ == "__main__":
    pass
