import cv2


RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

PARAMS = {
    "faces": {
        "scaleFactor": 1.3,
        "minNeighbors": 5,
    },
    "eyes": {
        "scaleFactor": 1.1,
        "minNeighbors": 22,
    },
    "smiles": {
        "scaleFactor": 1.7,
        "minNeighbors": 22,
    },
}


def detect(gray, frame):
    # Loading the cascades
    cascades_dir = "algorithms/resources/viola_jones/cascades"

    face_cascade = cv2.CascadeClassifier(f"{cascades_dir}/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(f"{cascades_dir}/haarcascade_eye.xml")
    smile_cascade = cv2.CascadeClassifier(f"{cascades_dir}/haarcascade_smile.xml")

    faces = face_cascade.detectMultiScale(
        image=gray,
        scaleFactor=PARAMS["faces"]["scaleFactor"],
        minNeighbors=PARAMS["faces"]["minNeighbors"],
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), RED, 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            image=roi_gray,
            scaleFactor=PARAMS["eyes"]["scaleFactor"],
            minNeighbors=PARAMS["eyes"]["minNeighbors"],
        )
        smiles = smile_cascade.detectMultiScale(
            image=roi_gray,
            scaleFactor=PARAMS["smiles"]["scaleFactor"],
            minNeighbors=PARAMS["smiles"]["minNeighbors"],
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), GREEN, 2)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), BLUE, 2)

    return frame
