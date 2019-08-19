import numpy as np

from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def get_confusion_matrix(
    data_path: str,
    num_images: int,
    image_size: list,
    batch_size: int,
    gen: ImageDataGenerator,
    model: Model,
):
    print(f"Generating confusion matrix: {num_images} images")
    predictions = []
    targets = []
    i = 0
    for x, y in tqdm(gen.flow_from_directory(
        data_path,
        target_size=image_size,
        shuffle=False,
        batch_size=batch_size * 2),
    ):

        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= num_images:
            break

    cm = confusion_matrix(targets, predictions)
    return cm
