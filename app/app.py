from app.workers.train import Train
from app.workers.predict import Predict
from app.config.config import Config
from app.config.train import Train as TrainConfigs
from app.config.work import Work as PredictConfigs


class Application:
    @staticmethod
    def run(work_type, config, work_dir):
        method = getattr(Application, work_type, None)
        if callable(method):
            method(config, work_dir)
        else:
            raise RuntimeError("Unsupported type detected")

    @staticmethod
    def train(config, work_dir):
        Train.run(TrainConfigs(Config(config)), work_dir=work_dir)

    @staticmethod
    def colorize(config, work_dir):
        """
            There will be shortcut to predict for our NN
            --continue=true
            --model_save=false
            --model_epochs=1
            --images_output_enable=true
            --images_output_step=1
        """
        config.__setattr__('continue', False)
        config.__setattr__('model_save', False)
        config.__setattr__('model_epochs', 1)
        config.__setattr__('images_output_enable', True)
        config.__setattr__('images_output_step', 1)
        Train.run(TrainConfigs(Config(config)), work_dir=work_dir, predict=True)
