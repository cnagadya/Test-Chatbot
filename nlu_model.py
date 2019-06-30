from rasa_nlu.converters import load_data
from rasa_nlu.config import RasaNLUConfig  # load config file
from rasa_nlu.model import Trainer


def train_nlu(data, config, model_dir):
    training_data = load_data(data)
    trainer = Trainer(RasaNLUConfig(config))
    trainer.train(training_data)

    model_directory = trainer.persist(model_dir, fixed_model_name="nlu")


if __name__ == "__main__":
    train_nlu("./data/data.json", "config.json", "./models/nlu")
