from face_recognition.dao import Connector

if __name__ == '__main__':
    connector = Connector()
    name_list, X = connector.load_embeddings([1, 2, 45, 100])
    print(len(name_list))
    print(name_list)
    print(X[0].shape)
    # print(X[0])