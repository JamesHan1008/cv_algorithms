import os
import cv2
import numpy as np
import tensorflow as tf

from .common.label_map_util import load_labelmap
from .common.visualization_util import visualize_boxes_and_labels_on_image_array

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = f"{DIR_PATH}/resources/ssd"
MODEL_NAME = "ssd_mobilenet_v2_oid_v4_2018_12_12"
LABEL_SET = "kuzushiji"

# Path to frozen detection graph, which contains the model that is used
PATH_TO_CKPT = os.path.join(MODEL_PATH, MODEL_NAME, LABEL_SET, "frozen_inference_graph.pb")

# Path to label map file
PATH_TO_LABELS = os.path.join(MODEL_PATH, MODEL_NAME, LABEL_SET, "label_map.pbtxt")

# Number of classes the object detector can identify
NUM_CLASSES = 6

# Load the label map
label_map = load_labelmap(PATH_TO_LABELS)

# Load the TensorFlow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    detection_graph_def = tf.GraphDef()

    with tf.gfile.GFile(PATH_TO_CKPT, "rb") as f:
        serialized_graph = f.read()
        detection_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(detection_graph_def, name="")

    sess = tf.Session(graph=detection_graph)


# Define tensors (i.e. data) for the object detection classifier

# Define the input tensor (image)
image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")

# Define the output tensors (detection boxes, scores, classes, and number of detections)
detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
num_detections = detection_graph.get_tensor_by_name("num_detections:0")


def detect(gray, frame):
    # Expand frame dimensions to have shape: [1, None, None, 3]
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    boxes, scores, classes, num = sess.run(
        [
            detection_boxes,
            detection_scores,
            detection_classes,
            num_detections,
        ],
        feed_dict={
            image_tensor: frame_expanded
        }
    )

    # Visualize detection results
    visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        label_map,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.45,
    )

    return frame


def webcam_detect():
    """
    This function is used for running the object detection algorithm on webcam feed directly inside
    this module for debugging purposes.
    """
    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        frame = detect(gray=None, frame=frame)
        cv2.imshow("Object detector", frame)

        # Press "q" to quit
        if cv2.waitKey(1) == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    webcam_detect()
