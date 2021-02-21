import psycopg2
import yaml
import pickle
import numpy as np
from typing import Optional, List, Tuple

from ..utils import Cfg


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

    def load_embeddings(self, ids_list: Optional[List[int]] = None) -> Tuple[List[str], np.ndarray]:
        cursor = self.connection.cursor()
        name_list = list()
        X = list()

        try:
            if ids_list is None:
                query = 'SELECT name, embeddings FROM datasets'
                cursor.execute(query)
            else:
                query = 'SELECT name, embeddings FROM datasets WHERE id IN %s'
                cursor.execute(query, (tuple(ids_list),))
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

        rows = cursor.fetchall()
        for row in rows:
            name = row[0]
            embedding = pickle.loads(row[1])
            for i in range(embedding.shape[0]):
                name_list.append(name)
            X.append(embedding)

        # Convert list of ndarray to ndarray
        X = np.concatenate(X, axis=0)
        cursor.close()

        if ids_list is not None:
            cursor = self.connection.cursor()
            query = 'SELECT DISTINCT id FROM datasets'
            cursor.execute(query)
            all_ids = cursor.fetchall()
            all_ids = [r[0] for r in all_ids]
            for id in ids_list:
                if id not in all_ids:
                    print('ID {} is not exist in database'.format(id))

            cursor.close()

        return name_list, X


if __name__ == '__main__':
    connector = Connector()
    name_list, X = connector.load_embeddings()
    print(name_list)
    print(X[0])


