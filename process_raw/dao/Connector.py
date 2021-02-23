import psycopg2
from face_recognition.utils import Cfg


class Singleton:
    def __init__(self, cls):
        self._cls = cls

    def get_instance(self):
        try:
            return self._instance._connection
        except AttributeError:
            self._instance = self._cls()
            return self._instance._connection

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')


@Singleton
class DBConnector:
    def __init__(self):
        # load database information
        db_config = Cfg.load_config()['postgres_db']

        self._connection = psycopg2.connect(**db_config)
        self._connection.set_session(autocommit=True)
        print("Successful connection")
        # self._init_database()
        # print('Successful init')

    def _init_database(self):
        path = Cfg.load_config()['init_db']
        with open(path, 'r') as f:
            query = f.read()
        cursor = self._connection.cursor()
        try:
            cursor.execute(query)
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            cursor.close()

    def __del__(self):
        print("Close connection")
        self._connection.close()


if __name__ == '__main__':
    connection1 = DBConnector.get_instance()
    connection2 = DBConnector.get_instance()
    print(connection1 is connection2)
