from flask import Flask, request, send_file
import warnings
# from black import main
import os
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import time
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import fire
from deepface.basemodels import (
    VGGFace,
    OpenFace,
    Facenet,
    Facenet512,
    FbDeepFace,
    DeepID,
    DlibWrapper,
    ArcFace,
    Boosting,
    SFaceWrapper,
)
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst
import tensorflow as tf
import cv2
import urllib
import numpy as np
import urllib.request


warnings.filterwarnings("ignore")

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
    import logging

    tf.get_logger().setLevel(logging.ERROR)


app = Flask(__name__)


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


# ESSENTIAL FUNCTION
def get_embeddings(
        img1_path,
        img2_path="",
        model_name="VGG-Face",
        distance_metric="cosine",
        model=None,
        enforce_detection=True,
        detector_backend="opencv",
        align=True,
        prog_bar=True,
        normalization="base",
):
    """
    This function verifies an image pair is same person or different persons.

    Parameters:
            img1_path, img2_path: exact image path, numpy array (BGR) or based64 encoded images could be passed. If you are going to call verify function for a list of image pairs, then you should pass an array instead of calling the function in for loops.

            e.g. img1_path = [
                    ['img1.jpg', 'img2.jpg'],
                    ['img2.jpg', 'img3.jpg']
            ]

            model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace or Ensemble

            distance_metric (string): cosine, euclidean, euclidean_l2

            model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times.

                    model = DeepFace.build_model('VGG-Face')

            enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.

            detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib

            prog_bar (boolean): enable/disable a progress bar

    Returns:
            Verify function returns a dictionary. If img1_path is a list of image pairs, then the function will return list of dictionary.

            {
                    "verified": True
                    , "distance": 0.2563
                    , "max_threshold_to_verify": 0.40
                    , "model": "VGG-Face"
                    , "similarity_metric": "cosine"
            }

    """

    tic = time.time()

    img_list, bulkProcess = functions.initialize_input(img1_path, img2_path)

    resp_objects = []
    embeddings = []

    # --------------------------------

    if model_name == "Ensemble":
        model_names = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]
        metrics = ["cosine", "euclidean", "euclidean_l2"]
    else:
        model_names = []
        metrics = []
        model_names.append(model_name)
        metrics.append(distance_metric)

    # --------------------------------

    if model == None:
        if model_name == "Ensemble":
            models = Boosting.loadModel()
        else:
            model = build_model(model_name)
            models = {}
            models[model_name] = model
    else:
        if model_name == "Ensemble":
            Boosting.validate_model(model)
            models = model.copy()
        else:
            models = {}
            models[model_name] = model

    # ------------------------------

    disable_option = (False if len(img_list) > 1 else True) or not prog_bar

    pbar = tqdm(range(0, len(img_list)), desc="Verification", disable=disable_option)

    for index in pbar:

        instance = img_list[index]

        if type(instance) == list and len(instance) >= 2:
            img1_path = instance[0]
            # img2_path = instance[1]

            ensemble_features = []

            for i in model_names:
                custom_model = models[i]

                # img_path, model_name = 'VGG-Face', model = None, enforce_detection = True, detector_backend = 'mtcnn'
                img1_representation = represent(
                    img_path=img1_path,
                    model_name=model_name,
                    model=custom_model,
                    enforce_detection=enforce_detection,
                    detector_backend=detector_backend,
                    align=align,
                    normalization=normalization,
                )
                # print("Shape of #1 embedding:",len(img1_representation))

                # img2_representation = represent(
                #     img_path=img2_path,
                #     model_name=model_name,
                #     model=custom_model,
                #     enforce_detection=enforce_detection,
                #     detector_backend=detector_backend,
                #     align=align,
                #     normalization=normalization,
                # )
                # print("Shape of #2 embedding:",len(img2_representation))
                embeddings.append(img1_representation)
                # embeddings.append(img2_representation)
        else:
            raise ValueError("Invalid arguments passed to verify function: ", instance)

    # -------------------------

    toc = time.time()

    embeddings = np.array(embeddings)
    return embeddings


def build_model(model_name):
    """
    This function builds a deepface model
    Parameters:
            model_name (string): face recognition or facial attribute model
                    VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
                    Age, Gender, Emotion, Race for facial attributes

    Returns:
            built deepface model
    """

    global model_obj  # singleton design pattern

    models = {
        "VGG-Face": VGGFace.loadModel,
        "OpenFace": OpenFace.loadModel,
        "Facenet": Facenet.loadModel,
        "Facenet512": Facenet512.loadModel,
        "DeepFace": FbDeepFace.loadModel,
        "DeepID": DeepID.loadModel,
        "Dlib": DlibWrapper.loadModel,
        "ArcFace": ArcFace.loadModel,
        "SFace": SFaceWrapper.load_model,
        "Emotion": Emotion.loadModel,
        "Age": Age.loadModel,
        "Gender": Gender.loadModel,
        "Race": Race.loadModel,
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj.keys():
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
            # print(model_name," built")
        else:
            raise ValueError("Invalid model_name passed - {}".format(model_name))

    return model_obj[model_name]


def represent(
        img_path,
        model_name="VGG-Face",
        model=None,
        enforce_detection=True,
        detector_backend="opencv",
        align=True,
        normalization="base",
):
    """
    This function represents facial images as vectors.

    Parameters:
            img_path: exact image path, numpy array (BGR) or based64 encoded images could be passed.

            model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace.

            model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times. Consider to pass model if you are going to call represent function in a for loop.

                    model = DeepFace.build_model('VGG-Face')

            enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.

            detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib

            normalization (string): normalize the input image before feeding to model

    Returns:
            Represent function returns a multidimensional vector. The number of dimensions is changing based on the reference model. E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
    """

    if model is None:
        model = build_model(model_name)

    # ---------------------------------

    # decide input shape
    input_shape_x, input_shape_y = functions.find_input_shape(model)

    # detect and align
    img = functions.preprocess_face(
        img=img_path,
        target_size=(input_shape_y, input_shape_x),
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
    )

    # ---------------------------------
    # custom normalization

    img = functions.normalize_input(img=img, normalization=normalization)

    # ---------------------------------

    # represent
    embedding = model.predict(img)[0].tolist()

    return embedding


@app.route('/', methods=['POST'])
def get_embedding_by_url():
    # url_tarantino = "https://img.20mn.fr/8aUPynXMTlCiD9wgR4lN7yk/1200x768_realisateur-quentin-tarantino"
    # url_tarantino2 = "https://media.npr.org/assets/artslife/movies/2009/08/tarantinofa-3b0dc6eaf870250bc12b177cf7543928bd897725.jpg"
    #
    # img_tarantino = url_to_image(url_tarantino)
    # img_tarantino2 = url_to_image(url_tarantino2)

    data = request.json
    img_link = data['link']
    # print(img_link)
    img = url_to_image(img_link)
    # print(img)

    # SHOW IMG BY URL
    # cv2.imshow("Image", img_tarantino)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    # img1 = img_tarantino2
    # img2 = img_tarantino

    # OLEG
    # img1=cv2.imread('/home/besmaks/Pictures/Oleg.jpg')
    # img1=cv2.imread('/home/besmaks/Pictures/Oleg2.jpg')
    # img2=cv2.imread('/home/besmaks/Pictures/Oleg2.jpg')

    # MAKS
    # img1=cv2.imread('/home/besmaks/Pictures/Maks1.jpg')
    # img2=cv2.imread('/home/besmaks/Pictures/Maks2.jpg')
    # img1=cv2.imread('/home/besmaks/Pictures/Maks3.jpg')

    # NIKITA
    # img1 = cv2.imread('/home/besmaks/Pictures/Nikitos.png')
    # img2 = cv2.imread('/home/besmaks/Pictures/Nikitos.png')

    # VOVA
    # img2 = cv2.imread('/home/besmaks/Pictures/Vova.png')

    # RESIZE
    new_width = 800
    new_height = 800

    dsize = (new_width, new_height)

    img1_resized = cv2.resize(img, dsize)
    # print(img1_resized)
    # img2 = cv2.resize(img2, dsize, interpolation=cv2.INTER_AREA)

    result = {'embeddings': get_embeddings(img1_resized).tolist()}
    # print(result)

    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
