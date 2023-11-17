from deepface import DeepFace
from sklearn.preprocessing import Normalizer
import joblib


class FaceRecognition:
    def __init__(self):
        self.model = joblib.load('utils/model.pkl')
        # self.encoder = Normalizer(norm='l2')
        self.label_encoder = joblib.load('utils/label_encoder.pkl')

    def recognize(self, image_arr):
        try:
            embedding = DeepFace.represent(image_arr, model_name='Facenet', detector_backend='mtcnn')[0]['embedding']
        except:
            return
        yhat_class = self.model.predict([embedding])
        yhat_proba = self.model.predict_proba([embedding])
        if yhat_proba[0][yhat_proba.argmax()] < 0.95:
            return
        label = self.label_encoder.inverse_transform(yhat_class)
        return label[0]
