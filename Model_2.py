#MODEL 2 WITH SVC
from Paths import path_to_save_downloaded_images
from sklearn.svm import SVC
from Training_set import Prepare_trainig_set
from Fetching_video import Video_to_test_embedds

class SVM_model:

    def __init__(self):
        pass

    def SVM(self):
        X, labels = Prepare_trainig_set.get_training_set_with_labels()
        test_face_embeddings, test_faces, test_frames = Video_to_test_embedds.get_test_embeddings()
        model = SVC()
        model.fit(X,labels)
        final_prediction_svc = []
        for image in range(len(test_face_embeddings)):
            prediction = []
            for face in range(len(test_face_embeddings[image])):
                predic = model.predict(test_face_embeddings[image][face])
                prediction.append(predic)
            final_prediction_svc.append(prediction)

        return final_prediction_svc