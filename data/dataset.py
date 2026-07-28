from torch.utils.data import Dataset
import torch


class MNISTCSVDATASET(Dataset):
    def __init__(self, images, labels=None, augment=None, transform=None):
        self.image = images.reshape((-1, 28, 28))
        self.image = torch.FloatTensor(self.image).unsqueeze(1)
        self.label = torch.LongTensor(labels) if labels is not None else None

        self.augment_transform = augment
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]

        if self.augment_transform:
            image = self.augment_transform(image)
        if self.transform:
            image = self.transform(image)

        if self.label is not None:
            return image, self.label[idx]
        return image


if __name__ == "__main__":
    pass
