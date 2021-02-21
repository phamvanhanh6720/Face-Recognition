import yaml
try:
    from importlib import resources
except ImportError:
    import importlib_resources as resources


class Cfg(dict):
    def __init__(self, config):
        super(Cfg, self).__init__(**config)
        self.__dict__ = self

    @staticmethod
    def load_config():
        """
        Load config from config.yml file in face_recognition package
        Returns: Dict

        """
        with resources.open_text('process_raw', 'config.yml') as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        return Cfg(cfg)