import requests
import os
import json
import cv2
from mtcnn import MTCNN
import numpy as np 


class LifeDetector:

    def __init__(self):
        self.server_url = os.getenv('LIFE_DETECTOR_ENDPOINT')
        self.detector = MTCNN()


    def __get_eye__(self, image):
        results = self.detector.detect_faces(image)
        if len(results)== 0:
            return
        square_size = int(results[0]['box'][2] * 0.35)

        left_side_eye = results[0]['keypoints']['left_eye'][1] - square_size // 2
        right_side_eye = results[0]['keypoints']['left_eye'][1] - square_size // 2 + square_size

        top_side_eye = results[0]['keypoints']['left_eye'][0] - square_size // 2
        bottom_side_eye = results[0]['keypoints']['left_eye'][0] - square_size // 2 + square_size

        left_eye = image[left_side_eye:right_side_eye, top_side_eye:bottom_side_eye]

        rescaled_image = cv2.resize(left_eye, (26, 34), interpolation=cv2.INTER_LINEAR)

        gray_image = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)
        gray_image = np.expand_dims(gray_image, axis=-1)

        return gray_image


    def detect(self, image):
        eye = self.__get_eye__(image)
        if eye is None:
            return
        data = json.dumps({"signature_name": "serving_default", "instances":[eye.tolist()]})
        response = requests.post(self.server_url, data=data)
        if response.status_code == requests.status_codes.codes.ALL_OK:
            result = json.loads(response.text)['predictions']
            return result[0][0]