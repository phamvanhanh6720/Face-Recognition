from queue import Queue
import math
import pickle
from typing import Tuple, Optional
from databases import Database


def track_queue(face_queue: Queue, queue_size=7) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Args:
        face_queue: Contain some elements: {label: str, spoof: int}
        # spoof: 0-fake 2d, 1-real, 2-fake 3d
        # special case: {label:"None", spoof: None} => Frame doesn't contain face

    Returns:
        Tuple (True/False, 'real'/'fake'/None, label/None)
    """

    elements = list(face_queue.queue)
    # print(elements)
    # key: label, value: list contain 3 values: [num_existence in queue,num_real, num_fake]
    unique_label = dict()
    unique_label["None"] = [0, 0, 0]

    for e in elements:
        l = e['label']
        spoof = e['spoof']

        if l not in unique_label.keys():
            unique_label[l] = [1, 1, 0] if 1 == spoof else [1, 0, 1]

        elif l != "None":
            num_real = unique_label[l][1]
            num_fake = unique_label[l][2]
            num_existence = unique_label[l][0]

            if spoof == 1:
                unique_label[l] = [num_existence + 1, num_real + 1, num_fake]
            else:
                unique_label[l] = [num_existence + 1, num_real, num_fake + 1]
        else:
            temp = unique_label["None"][0]
            unique_label["None"] == [temp + 1, 0, 0]

    # Voting
    max_existence = -1
    name = None
    for key in unique_label.keys():
        if max_existence < unique_label[key][0]:
            name = key
            max_existence = unique_label[key][0]

    if name != "None":
        if max_existence < math.ceil(queue_size / 2):
            return (False, None, None)

        # num_fake >= num_real
        if unique_label[name][2] >= unique_label[name][1]:
            return (True, 'fake', name)

        return (True, 'real', name)

    return (False, None, None)


async def store_image(database: Database, room_id, student_id, image):

    query = """INSERT INTO get_info(Room_id, Student_id, Image) VALUES(:room_id, :student_id, :image)"""
    values = {"room_id": room_id, "student_id": student_id, "image": pickle.dumps(image)}
    await database.execute(query=query, values=values)

