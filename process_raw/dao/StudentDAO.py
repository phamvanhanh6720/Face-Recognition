from process_raw.dao.Connector import DBConnector
from face_recognition.dao import Student
import psycopg2
import pickle
import numpy as np
from typing import Optional, List, Tuple


class StudentDAO:
    def __init__(self):
        self._connection = DBConnector.get_instance()

    def store_one_person(self, student: Student):

        cursor = self._connection.cursor()
        query = """INSERT INTO datasets(id, name, cropped_images, embeddings)
                    VALUES (%s, %s, %s, %s)"""

        try:
            cursor.execute(query, (
            student.student_id, student.name, pickle.dumps(student.cropped_images), pickle.dumps(student.embeddings)))
            print("Store successful")

        except:
            print("ID = {} already exists. Please change ID".format(id))

        cursor.close()

if __name__ == '__main__':
    pass
