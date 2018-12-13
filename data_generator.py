import pandas as pd
import numpy as np
from keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, dataset: pd.DataFrame, dim: list, batch_size: int):
        self.dataset: pd.DataFrame = dataset
        self.dim = dim
        self.batch_size = batch_size

    @staticmethod
    def generate_batch(df: pd.DataFrame, batch_size: int, dim_0: int, dim_1: int, current_batch: int=0):
        start_idx = batch_size * current_batch
        X = np.empty((batch_size, dim_0, dim_1))
        y = np.empty(batch_size, dtype=int)

        for i in range(batch_size):
            pos = df.iloc[start_idx + i]
            X[i, ] = np.load(pos.spectrogram_path)
            y[i] = pos.speaker_id_encoded
        return X, y

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, current_batch):
        return self.generate_batch(self.dataset, self.batch_size, self.dim[0], self.dim[1], current_batch)

    def on_epoch_end(self):
        self.dataset = self.dataset.sample(frac=1)
