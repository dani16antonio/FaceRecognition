# FaceRecognition
Face Recognition Pipeline System 
Architecture based on Jason Brownlee [blog](https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/) if you want to know more about this pipeline I strongly suggest you take a look.
In order to test this system, you can to use the Kaggle dataset [5 celebrity faces](https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset).
## Architecture 
This architecture has 3 blocks or modules, each of one has and specific task:
* **First block**: this is the **face detection** block, its task is to detect faces in the input image, and save it, in order to perform this task, we use Multi-Task Cascaded Convolutional Neural Network, or MTCNN.
* **Second block**: this is the **embeddings** block, its taks is to process the faces detected on the previous block and get its embedding vector, in orden to perform this task, we use FaceNet Keras' implemention.
* **Third block**: this is the recognition block, the input of this this block is the embeddings and perform Support Vector Machine, or SVM to recognize.
<img src="resources/architecture diagram.png">