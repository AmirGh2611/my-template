import torch
from torchvision import transforms

TRAIN_PATH = r"D:/Dataset/train.csv"
TEST_PATH = r"D:/Dataset/test.csv"
DATA_AUGMENTATION = transforms.Compose(
    [
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
)
DATA_NORMALIZATION = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRIALS = 5
MAX_EPOCHS_PER_TRIAL = 10
EARLY_STOP_PATIENCE = 3
STUDY_NAME = "mnist_cnn_search"
STUDY_STORAGE = "sqlite:///outputs/optuna_study.db"
FINAL_MAX_EPOCHS = 30
if __name__ == "__main__":
    pass
