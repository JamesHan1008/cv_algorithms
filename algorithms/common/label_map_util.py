# Some functions of this module are copies or modified versions from the TensorFlow Object
# Detection API: https://github.com/tensorflow/models/tree/master/research/object_detection

import logging

from typing import List

import tensorflow as tf

from google.protobuf import text_format
from .protos import string_int_label_map_pb2


def _validate_label_map(label_map: string_int_label_map_pb2.StringIntLabelMap):
    """
    Checks if a label map is valid
    :param label_map: label map proto
    """
    for item in label_map.item:
        if item.id < 0:
            raise ValueError("Label map ids should be >= 0.")
        if (item.id == 0 and item.name != "background" and
                item.display_name != "background"):
            raise ValueError("Label map id 0 is reserved for the background label")


def _convert_label_map_to_categories(
        label_map: string_int_label_map_pb2.StringIntLabelMap,
        max_num_classes: int,
        use_display_name: bool = True
) -> List[dict]:
    """
    Convert a label map proto into a dictionary of COCO compatible categories keyed by category id
    :param label_map: label map proto
    :param max_num_classes:
    :param use_display_name: whether or not to load "display_name" as category name
    :return: a dictionary keyed by the object id, where each element is a dictionary of id (int) and name (str)
    """
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
            categories.append({
                "id": class_id + label_id_offset,
                "name": "category_{}".format(class_id + label_id_offset)
            })
        return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info(
                "Ignore item %d since it falls outside of requested "
                "label range.", item.id)
            continue
        if use_display_name and item.HasField("display_name"):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({"id": item.id, "name": name})

    category_index = {}
    for cat in categories:
        category_index[cat["id"]] = cat

    return category_index


def load_labelmap(path: str) -> List[dict]:
    """
    Loads label map proto
    :param path: path to StringIntLabelMap proto text file
    :return: a dictionary of COCO compatible categories keyed by category id
    """
    with tf.gfile.GFile(path) as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)

    _validate_label_map(label_map)

    label_map = _convert_label_map_to_categories(label_map, max_num_classes=len(label_map.item))

    return label_map
