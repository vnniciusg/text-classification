if __name__ == "__main__":

    __import__("warnings").filterwarnings("ignore")
    __import__("os").environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    from loguru import logger

    from config import Config
    from data.data_loader import IMDBReviewsDataLoader
    from model.rnn_model import RNNTextClassifierModel, RNNTextClassifierModelTrainer

    config = Config()

    data_loader = IMDBReviewsDataLoader(config=config)

    model = RNNTextClassifierModel(config, data_loader.train_dataset)
    trainer = RNNTextClassifierModelTrainer(model, data_loader.train_dataset, data_loader.test_dataset, config)

    history = trainer.train()
    test_loss, test_acc = model.evaluate(data_loader.test_dataset)

    logger.info("test Loss:", test_loss)
    logger.info("test Accuracy:", test_acc)
