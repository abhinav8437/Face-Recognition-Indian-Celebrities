from sklearn.neighbors import KNeighborsClassifier
import pickle as pk
from Paths import celeb_list
import Training_set
import cv2
import os

class KNN_model:

    def __init__(self):
        pass

    def KNN(self):
        X,labels,pure_face_embedds = Training_set.Prepare_trainig_set(celeb_list,num_of_pics_for_each_celeb=95).get_training_set_with_labels()
        # test_face_embeddings, test_faces, test_frames = Video_to_test_embedds().get_test_embeddings()
        os.chdir("/Users/abhinavrohilla/video_images/")
        # pk.dump(test_face_embeddings,open("test_face_embeddings.p","wb"))
        test_face_embeddings = pk.load(open("test_face_embeddings.p","rb"))
        # pk.dump(test_frames, open("test_frames.p", "wb"))
        test_frames = pk.load(open("test_frames.p", "rb"))
        knn_obj = KNeighborsClassifier(n_neighbors=13)
        knn_obj.fit(X,labels)
        final_prediction_svc = []
        for image in range(len(test_face_embeddings)):
            prediction = []
            for face in range(len(test_face_embeddings[image])):
                predic = knn_obj.predict(test_face_embeddings[image][face])
                prediction.append(predic)
            final_prediction_svc.append(prediction)

        return final_prediction_svc,test_frames


    def generate_box_and_name(self, imagess, texts):
        face_cascade = cv2.CascadeClassifier(
            '/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
        for image in range(len(imagess)):
            faces_ = face_cascade.detectMultiScale(imagess[image], 1.3, 5)
            dict_faces = {}
            for num, face in enumerate(faces_):
                dict_faces[num] = face
            for i, face in enumerate(dict_faces):
                for (x, y, w, h) in [dict_faces[face]]:
                    img = cv2.rectangle(imagess[image], (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(imagess[image], texts[image][i][0], (x, y), cv2.FONT_HERSHEY_PLAIN, 3.5,
                                (255, 0, 0), 2)

    def make_video_from_frames(self, test_frames, final_prediction):
        self.generate_box_and_name(test_frames, final_prediction)
        frame_width, frame_height = test_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('outputttt.avi', fourcc, 20.0, (frame_width, frame_height))
        for i in range(len(test_frames)):
            frame = test_frames[i]
            # write the flipped frame
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()



pred,test_frames= KNN_model().predict()
KNN_model().make_video_from_frames(test_frames,pred)
