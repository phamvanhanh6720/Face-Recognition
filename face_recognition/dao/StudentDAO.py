from face_recognition.dao.Connector import DBConnector
import psycopg2
import pickle
import numpy as np
from typing import Optional, List, Tuple


class Student:
    def __init__(self, student_id=None, name=None, cropped_images=None, embeddings=None):
        self._student_id = student_id
        self._name = name
        self._cropped_images = cropped_images
        self._embeddings = embeddings

    @property
    def student_id(self):
        return self._student_id

    @student_id.setter
    def student_id(self, value):
        self._student_id = value

    @student_id.deleter
    def student_id(self):
        del self._student_id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @name.deleter
    def name(self):
        del self._name

    @property
    def cropped_images(self):
        return self._cropped_images

    @cropped_images.setter
    def cropped_images(self, value):
        self._cropped_images = value

    @cropped_images.deleter
    def cropped_images(self):
        del self._cropped_images

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value):
        self._embeddings = value

    @embeddings.deleter
    def embeddings(self):
        del self._embeddings


class StudentDAO:
    def __init__(self):
        self._connection = DBConnector.get_instance()

    def load_embeddings(self, ids_list: Optional[List[int]] = None) -> Tuple[List[str], np.ndarray]:
        cursor = self._connection.cursor()
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
    connection = DBConnector.get_instance()
    # name_list, X = connection.load_embeddings()
    # print(name_list)
    # print(X[0])

