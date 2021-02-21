import psycopg2
import pickle
import numpy as np
from typing import Optional, List

from process_raw.utils import Cfg


class Connector():
    def __init__(self):
        # load database information
        db_config = Cfg.load_config()['postgres_db']

        self.connection = psycopg2.connect(**db_config)
        self.connection.set_session(autocommit=True)
        print("Successful connection")

    def __del__(self):
        print("Close connection")
        self.connection.close()

    def store_one_person(self, id: int, name:str, cropped_images: List[np.ndarray],
                         embeddings: np.ndarray):

        cursor = self.connection.cursor()
        query = """INSERT INTO datasets(id, name, cropped_images, embeddings)
                    VALUES (%s, %s, %s, %s)"""

        try:
            cursor.execute(query, (id, name, pickle.dumps(cropped_images), pickle.dumps(embeddings)))
            print("Store successful")

        except:
            print("ID = {} already exists. Please change ID".format(id))

        cursor.close()


if __name__ == '__main__':
    connector = Connector()
    raw_images = [np.arange(6).reshape(2,3)]
    cropped_images = [np.arange(6).reshape(2,3)]
    embeddings = np.random.rand(3, 512)

    connector.store_one_person(id=2, name='phạm văn hạnh', raw_images=raw_images, cropped_images=cropped_images, embeddings=embeddings)


