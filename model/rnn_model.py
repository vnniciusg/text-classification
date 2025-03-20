from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import TextVectorization, Embedding, Bidirectional, Dense, LSTM

from config import Config
from utils import timeit


class RNNTextClassifierModel(Model):

    def __init__(self, config: Config, train_dataset=None) -> None:
        super().__init__()
        self.config = config

        self.encoder = TextVectorization(max_tokens=self.config.VOCAB_SIZE)
        if train_dataset:
            self.encoder.adapt(train_dataset.map(lambda text, label: text))

        self.embedding = Embedding(input_dim=len(self.encoder.get_vocabulary()), output_dim=64, mask_zero=True)

        self.bidirectional_1 = Bidirectional(LSTM(64, return_sequences=True))
        self.bidirectional_2 = Bidirectional(LSTM(32))
        self.dense = Dense(64, activation="relu")
        self.dense_output = Dense(1)

    def __call__(self, inputs):
        x = self.encoder(inputs)
        x = self.embedding(x)
        x = self.bidirectional_1(x)
        x = self.bidirectional_2(x)
        x = self.dense(x)
        return self.dense_output(x)


class RNNTextClassifierModelTrainer:

    def __init__(self, model: RNNTextClassifierModel, train_dataset, val_dataset, config: Config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.optmizer = Adam(self.config.TRAINING_LEARNING_RATE)

    @timeit
    def train(self):

        self.model.compile(optimizer=self.optmizer, loss=BinaryCrossentropy(from_logits=True), metrics=["accuracy"])

        return self.model.fit(self.train_dataset, epochs=self.config.EPOCHS, validation_data=self.val_dataset, validation_steps=30)
