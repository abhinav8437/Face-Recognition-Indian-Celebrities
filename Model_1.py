import numpy as np
import Training_set
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
import pickle as pk
import cv2
from Paths import celeb_list
import os

class cosine_or_eucledian_distance():

    def __init__(self):
        self.celeb_label_with_embeddings = None
        self.celeb_embed_mean = None

    #IN THIS DICTIONARY EVERY CELEBRITY HAS A AVERAGE MEAN OF THEIR FACES, WHICH YOU CAN SAY THE PROFILE FOR EACH CELEB
#IN THIS MEAN OF EACH CELEB IS THE PROFILE OF OF EACH CELEB
    # IN THIS DICTIONARY EVERY CELEBRITY HAS A AVERAGE MEAN OF THEIR FACES, WHICH YOU CAN SAY THE PROFILE FOR EACH CELEB
    # IN THIS MEAN OF EACH CELEB IS THE PROFILE OF OF EACH CELEB
    def profile_of_each_celeb(self,celeb_label_with_embeddings, celeb_list):
        celeb_embed_mean = {}
        for i in range(len(celeb_label_with_embeddings)):
            celeb_embed_mean[celeb_list[i]] = np.mean(celeb_label_with_embeddings[celeb_list[i]], axis=0).reshape(1, -1)
        return celeb_embed_mean


    final_prediction = []
    def names_of_celeb_in_video(self,test_embedds, celeb_embed_mean):
        celebrities_in_pic = []
        cosine = []
        eucledian = []
        for celeb in celeb_list:
            #         aa = cosine_similarity(test_embedds,celeb_embed_mean[celeb])
            b = euclidean_distances(test_embedds, celeb_embed_mean[celeb])
            #         cosine.append(aa)
            eucledian.append(b)
        max_similar_star = np.argmin(eucledian, axis=0)
        #     max_similar_star = np.argmax(cosine,axis=0)
        #     print (np.max(cosine,axis=0))
        #     print (np.min(eucledian,axis=0))
        for i in max_similar_star:
            celebrities_in_pic.append(celeb_list[int(i)])
        return celebrities_in_pic

    def predict(self):
        X,labels,pure_face_embedds = Training_set.Prepare_trainig_set(celeb_list,num_of_pics_for_each_celeb=95).get_training_set_with_labels()

        celeb_embed_mean = self.profile_of_each_celeb(pure_face_embedds, celeb_list)
        test_face_embeddings, test_faces, test_frames = Video_to_test_embedds().get_test_embeddings()
        os.chdir("/Users/abhinavrohilla/video_images/")
        # pk.dump(test_face_embeddings,open("test_face_embeddings.p","wb"))
        test_face_embeddings = pk.load(open("test_face_embeddings.p", "rb"))
        final_prediction = []
        for image in range(len(test_face_embeddings)):
            prediction = []
            for face in range(len(test_face_embeddings[image])):
                predic = self.names_of_celeb_in_video(test_face_embeddings[image][face], celeb_embed_mean)
                prediction.append(predic)
            final_prediction.append(prediction)
        self.final_prediction = final_prediction
        return final_prediction,test_frames

    def generate_box_and_name(self,imagess, texts):
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
                    cv2.putText(imagess[image], texts[image][i][0], (x, y), cv2.FONT_HERSHEY_PLAIN, 3.5, (255, 0, 0), 2)

    def make_video_from_frames(self,test_frames,final_prediction):
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


pred,test_frames= cosine_or_eucledian_distance().predict()
cosine_or_eucledian_distance().make_video_from_frames(test_frames,pred)

