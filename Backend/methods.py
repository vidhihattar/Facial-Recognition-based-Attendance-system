import numpy as np
from deepface import DeepFace
import os
from tqdm import tqdm
from deepface.commons import functions
import pandas as pd

model = DeepFace.build_model("Facenet")


def calculateEmbedding(facial_img_path):
    facial_img_temp = functions.extract_faces(
        facial_img_path, target_size=(160, 160), enforce_detection=False)
    facial_img = facial_img_temp[0][:-2]
    # represent
    embedding = model.predict(facial_img[0])[0].tolist()
    return embedding


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(
        euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def searchPeople(image):
    facial_img_temp = functions.extract_faces(image, target_size=(
        160, 160), detector_backend="retinaface", enforce_detection=True)
    groupEmb = []
    for face in facial_img_temp:
        facial_img = face[:-2]
        # represent
        embedding = model.predict(facial_img[0])[0].tolist()
        groupEmb.append(embedding)

    return groupEmb
