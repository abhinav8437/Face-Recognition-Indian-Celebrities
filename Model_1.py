from Paths import celeb_list
from Training_set import Prepare_trainig_set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from Fetching_video import Video_to_test_embedds


class cosine_or_eucledian_distance():
    def __init__(self):
        self.celeb_label_with_embeddings = None
        self.celeb_embed_mean = None

    #IN THIS DICTIONARY EVERY CELEBRITY HAS A AVERAGE MEAN OF THEIR FACES, WHICH YOU CAN SAY THE PROFILE FOR EACH CELEB
#IN THIS MEAN OF EACH CELEB IS THE PROFILE OF OF EACH CELEB
    def profile_of_each_celeb(self):
        self.celeb_label_with_embeddings = Prepare_trainig_set.celeb_label_with_embeddings
        self.celeb_embed_mean = {}
        for i in range(len(self.celeb_label_with_embeddings)):
            self.celeb_embed_mean[celeb_list[i]] = np.mean(self.celeb_label_with_embeddings[celeb_list[i]],axis=0).reshape(1,-1)

    def names_of_celeb_in_video(self,test_embedds):
        celebrities_in_pic = []
        cosine = []
        eucledian = []

        for celeb in celeb_list:
    #         aa = cosine_similarity(test_embedds,self.celeb_embed_mean[celeb])
            b = euclidean_distances(test_embedds,self.celeb_embed_mean[celeb])
    #         cosine.append(aa)
            eucledian.append(b)
        max_similar_star = np.argmin(eucledian,axis=0)
    #     max_similar_star = np.argmax(cosine,axis=0)
    #     print (np.max(cosine,axis=0))
    #     print (np.min(eucledian,axis=0))
        for i in max_similar_star:
             celebrities_in_pic.append(celeb_list[int(i)])
        return celebrities_in_pic

    # MODEL_1 with COSINE AND EUCLEDIAN DISTANCE
    def predict(self):
        test_face_embeddings, test_faces, test_frames = Video_to_test_embedds.get_test_embeddings()

        final_prediction = []
        for image in range(len(test_face_embeddings)):
            prediction = []
            for face in range(len(test_face_embeddings[image])):
                predic = self.names_of_celeb_in_video(test_face_embeddings[image][face], self.celeb_embed_mean)
                prediction.append(predic)
            final_prediction.append(prediction)
        return final_prediction