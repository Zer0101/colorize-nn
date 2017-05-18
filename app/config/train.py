from app.config.config import Config


class Train:
    configs = {
        "log": {},
        "save": {},
        "learning_rate": {},
        "images": {
            "input": {},
            "output": {}
        }
    }

    def __init__(self, config):
        if not isinstance(config, Config):
            raise TypeError("This class accept only Config instances")

        self.configs['id'] = config.fetch(index='model_id')
        if self.configs['id'] is None:
            raise ValueError('Model ID is required')
        self.configs['dir'] = config.fetch(index='model_dir')
        if self.configs['dir'] is None:
            raise ValueError('Path to directory with models must be specified')
        self.configs['continue'] = bool(config.fetch(index='continue'))

        self.configs['log']['level'] = config.fetch(index='model_log_level')
        self.configs['log']['dir'] = config.fetch(index='model_log_dir')
        if self.configs['log']['level'] > 0 and self.configs['log']['dir'] is None:
            raise ValueError('If model logging is enabled log directory must be defined')

        self.configs['save']['enable'] = bool(config.fetch(index='model_save'))
        self.configs['save']['step'] = config.fetch(index='model_save_pass')
        self.configs['save']['path'] = config.fetch(index='model_save_path')
        self.configs['save']['format'] = config.fetch(index='model_save_path')
        if self.configs['save']['enable'] and self.configs['save']['path'] is None:
            raise ValueError('If model saving is enabled, save directory must be defined')

        self.configs['learning_rate']['value'] = config.fetch(index='model_learning_rate')
        self.configs['learning_rate']['step'] = config.fetch(index='model_learning_rate_step')
        self.configs['learning_rate']['decay'] = config.fetch(index='model_learning_rate_decay')

        self.configs['epochs'] = config.fetch(index='model_epochs')
        self.configs['images']['batch_size'] = config.fetch(index='images_batch_size')
        self.configs['images']['input']['path'] = config.fetch(index='images_input')
        if self.configs['images']['input']['path'] is None:
            raise ValueError('To train neural network model directory with training cases is required')
        self.configs['images']['output']['enable'] = bool(config.fetch(index='images_output_enable'))
        self.configs['images']['output']['step'] = config.fetch(index='images_output_step')
        self.configs['images']['output']['path'] = config.fetch(index='images_output')
        self.configs['images']['output']['format'] = config.fetch(index='images_output_format')
        if self.configs['images']['output']['enable'] and self.configs['images']['output']['path'] is None:
            raise ValueError('If images output enabled, directory for output is required')
