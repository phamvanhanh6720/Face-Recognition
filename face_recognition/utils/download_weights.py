import gdown
from pathlib import Path


def download_weights(url, cache=None, md5=None, quiet=False):

    return Path(gdown.cached_download(url, cache=cache, md5=md5, quiet=quiet))