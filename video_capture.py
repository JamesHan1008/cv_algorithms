import argparse
import os

import cv2

from algorithms.common.model_util import load_trained_model

DETECT_ALGORITHM = "tf_ssd"
CLASSIFY_ALGORITHM = "vgg16"

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_classify_function():
    if CLASSIFY_ALGORITHM == "resnet":
        from algorithms.resnet import classify
    elif CLASSIFY_ALGORITHM == "vgg16":
        from algorithms.vgg16 import classify
    else:
        raise ValueError(f"Invalid classification algorithm: {CLASSIFY_ALGORITHM}")
    return classify


def get_detect_function():
    if DETECT_ALGORITHM == "viola_jones":
        from algorithms.viola_jones import detect
    elif DETECT_ALGORITHM == "tf_ssd":
        from algorithms.tf_object_detection import detect
    else:
        raise ValueError(f"Invalid detection algorithm: {DETECT_ALGORITHM}")
    return detect


def object_classification():
    """ Demonstrate object classification algorithms using the webcam """
    classify = get_classify_function()

    # TODO: better way to handle loading models
    model = load_trained_model(f"{dir_path}/algorithms/resources/{CLASSIFY_ALGORITHM}/models/fruits_360.h5")
    if model is None:
        raise Exception("failed to load model")

    video_capture = cv2.VideoCapture(0)

    while True:
        _, frame = video_capture.read()  # X by Y by 3 (RGB)
        cv2.imshow("Video", frame)

        # Press "space" to classify the object in the webcam
        if cv2.waitKey(1) & 0xFF == ord(" "):
            X = cv2.resize(frame, (100, 100))
            label = classify(X, model)
            print(label)

        # Press "q" to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def object_detection():
    """ Demonstrate object detection algorithms using the webcam """
    detect = get_detect_function()

    video_capture = cv2.VideoCapture(0)

    while True:
        _, frame = video_capture.read()  # X by Y by 3 (RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect(gray, frame)
        cv2.imshow("Video", canvas)

        # Press "q" to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--detect", action="store_true", help="Object detection")
    parser.add_argument("-c", "--classify", action="store_true", help="Object classification")
    args = vars(parser.parse_args())

    if args["detect"]:
        object_detection()
    elif args["classify"]:
        object_classification()


if __name__ == "__main__":
    main()
