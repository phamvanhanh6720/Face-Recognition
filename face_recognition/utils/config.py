import yaml
import os
# from importlib import resources


class Cfg(dict):
    def __init__(self, config):
        super(Cfg, self).__init__(**config)
        self.__dict__=self

    @staticmethod
    def load_config():
        """
        Load config from config.yml file in face_recognition package
        Returns: Dict

        """
        with open(os.path.join('face_recognition', 'config.yml')) as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        return Cfg(cfg)