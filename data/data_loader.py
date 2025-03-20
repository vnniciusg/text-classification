from tensorflow.data import AUTOTUNE
import tensorflow_datasets as tfds

from config import Config


class IMDBReviewsDataLoader:

    def __init__(self, config: Config):
        self.config = config
        self.dataset, self.info = self._load_raw_data()
        self._split_train_test()

    def _load_raw_data(self) -> str:
        return tfds.load("imdb_reviews", with_info=True, as_supervised=True)

    def _split_train_test(self):
        self.train_dataset, self.test_dataset = self.dataset["train"], self.dataset["test"]
        self.train_dataset = self.train_dataset.shuffle(self.config.BUFFER_SIZE).batch(self.config.BATCH_SIZE).prefetch(AUTOTUNE)
        self.test_dataset = self.test_dataset.batch(self.config.BATCH_SIZE).prefetch(AUTOTUNE)
