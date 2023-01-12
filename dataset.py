import os
import pandas as pd
import cv2 as cv
from torch.utils.data import Dataset
import random
import numpy as np


class OCRDataset(Dataset):
    def __init__(
        self, annotations_file,
        img_dir, transformations=None,
        labels_transformations=None,
        start_at=0
    ):
        self.labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transformations = transformations
        self.labels_transformations = labels_transformations
        self.start_at = start_at

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if idx < self.start_at:
            return None, None
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0]) + '.png'
        image = cv.imread(img_path, 0)
        image = image / 255
        #image = 1 - image
        label = self.labels.iloc[idx, 1]

        images = [image]

        if self.transformations:
            for t in self.transformations:
                new_img = t(image)
                images.append(new_img)

        labels = [label for _ in range(len(images))]

        if self.labels_transformations:
            for t in self.labels_transformations:
                for i in range(len(labels)):
                    labels[i] = t(labels[i])

        random.shuffle(images)

        return np.array(images, dtype=image.dtype), np.array(labels, dtype=np.uint8)
