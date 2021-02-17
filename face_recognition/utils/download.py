import gdown
import os


def download_weights(url, cache=None, md5=None, quiet=False):

    return os.path.join(gdown.cached_download(url, path=cache, md5=md5, quiet=quiet))