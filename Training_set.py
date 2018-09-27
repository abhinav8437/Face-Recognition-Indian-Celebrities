import cv2
import os
import numpy as np
import pickle as pk
import sys
from sklearn.cluster import KMeans
# import imutils
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
import time
from parse_images_from_google import parse_images_of_celebrities_from_google
from Paths import celeb_list,path_to_save_video_frames
from sklearn.svm import SVC

class Prepare_trainig_set():

    def __init__(self):
        self.celeb_label_with_embeddings = None

    # RETURN IMAGE IN VECTOR FORM
    # RETURN IMAGE IN VECTOR FORM
    def image_to_vector(self,dataset="train"):
        bgr_images = []
        gray_scale_images = []
        not_recognized_images_count = 0
        labels = []
        if (dataset == "train"):
            for celeb in celeb_list:
                path = os.chdir('/Users/abhinavrohilla/Data/downloads/{}'.format(celeb))
                file_names = os.listdir()
                for image_name in range(len(file_names)):
                    path = '/Users/abhinavrohilla/Data/downloads/{}/{}'.format(celeb, file_names[image_name])
                    image = cv2.imread(path)

                    try:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        gray_scale_images.append(gray)
                        labels.append(celeb)
                        bgr_images.append(image)
                    except:
                        not_recognized_images_count = not_recognized_images_count + 1
                        image_name = image_name + 1

        elif (dataset == "test"):
            path = os.chdir(path_to_save_video_frames)
            total_images_in_folder = os.listdir()
            for image_name in range(len(total_images_in_folder)):
                path = '/Users/abhinavrohilla/Data/video_images/frame{}.jpg'.format(image_name)
                image = cv2.imread(path)

                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray_scale_images.append(gray)
                    labels.append([])
                    bgr_images.append(image)
                except:
                    not_recognized_images_count = not_recognized_images_count + 1
                    image_name = image_name + 1

        return np.array(gray_scale_images), np.array(bgr_images), labels

    # WORKS ONLY FOR GRAY IMAGES
    # THIS IS FOR CREATING TRAINING DATA FACES
    def detect_face_and_crop(self,gray_images_vec,labels):
        face_cascade = cv2.CascadeClassifier(
            '/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
        test_faces = []
        face_labels = []
        for image in range(gray_images_vec.shape[0]):
            dict_faces = {}
            faces_ = face_cascade.detectMultiScale(gray_images_vec[image], 1.3, 5)
            for num, dic_face in enumerate(faces_):
                dict_faces[num] = dic_face
            for face in dict_faces:
                for (x, y, w, h) in [dict_faces[face]]:
                    img = cv2.rectangle(gray_images_vec[image], (x, y), (x + w, y + h), (255, 0, 0), 2)
                    bounding_box = img[y:y + h, x:x + w]
                    try:
                        test_face = cv2.resize(bounding_box, (160, 160))
                        test_faces.append(test_face)
                        face_labels.append(labels[image])
                    except:
                        continue
        return np.array(test_faces), face_labels

    def vector_to_embeddings(self,faces):
        sys.path.append("/Users/abhinavrohilla")
        from facenet.src import facenet
        model = "/Users/abhinavrohilla/Downloads/VGGFace2 pretrained model"
        sess = tf.Session()
        with sess.as_default():
            facenet.load_model(model)
            image_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            prewhiten_face = facenet.prewhiten(faces)
            feed_dict = {image_placeholder: prewhiten_face, phase_train_placeholder: False}
            embedds = sess.run(embeddings, feed_dict=feed_dict)
        return embedds

    def remove_uncessary_faces_fetched_from_google(self,face_embeddings, top_sort):
        # CLEANING OTHER FACES FROM THE FACES FETCHED FROM GOOGLE
        # IN THIS MEAN IS TAKEN TO CLEAN OTHER FACES
        arg_list = []
        for i, celeb in enumerate(range(0, face_embeddings.shape[0], 94)):
            celebrity = np.mean(face_embeddings[celeb:celeb + 94], axis=0)
            cosine = cosine_similarity(celebrity.reshape(1, -1), face_embeddings)
            cosine = cosine.reshape(1, -1)
            cosine_sort = np.argsort(cosine).reshape(-1, 1)
            cosine_top = cosine_sort[::-1]
            arg_list.append(cosine_top[:top_sort])
        # THIS LIST WILL HAVE INDEX OF FACES WITH NO NOISE FROM FACESS ARRAY
        # 60 IMAGES PER CELEBRITY AND NO OTHER CELEBRITY WOULD BE PRESENT
        best_faces_of_celeb = np.array(arg_list).reshape(-1, top_sort)
        return best_faces_of_celeb

    def create_training_data_with_labels(self,celeb_list, face_vector, face_embeddings, top_sort,best_faces_of_celeb):
        # DICTIONARY OF EVERY ACTOR WITH THEIR FACES EMBEDDINGS
        celeb_label_with_embeddings = {}
        # LIST OF THOSE FACES VECTORS OF ABOVE EMBEDDINGS
        celeb_faces_vector = []
        for celeb in range(len(celeb_list)):
            celeb_label_with_embeddings[celeb_list[celeb]] = face_embeddings[best_faces_of_celeb][celeb]
            celeb_faces_vector.append(face_vector[best_faces_of_celeb[celeb]])
        celeb_faces_vector = np.array(celeb_faces_vector).reshape(top_sort * len(celeb_list), 160, 160)
        # RECOVER TRAIN DATA AND LABELS FROM DICTIONARY
        labels = []
        X = []
        for i in range(len(celeb_label_with_embeddings)):
            for j in range(len(celeb_label_with_embeddings[celeb_list[i]])):
                labels.append(celeb_list[i])
                X.append(celeb_label_with_embeddings[celeb_list[i]][j])
        X = np.array(X)
        return X, labels, celeb_faces_vector, celeb_label_with_embeddings

    def get_training_set_with_labels(self):
        parse_images_of_celebrities_from_google(celebrities=celeb_list,num_of_images=95)
        gray_images, bgr_images, labels = self.image_to_vector("train")
        facess, labels = self.detect_face_and_crop(gray_images)
        # FACENET REQUIRES BGR IMAGES
        faces_for_embeddings = np.array([cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in facess])
        face_embeddings = self.vector_to_embeddings(faces_for_embeddings)
        best_faces_of_celeb = self.remove_uncessary_faces_fetched_from_google(face_embeddings, top_sort=60)
        X, labels, celeb_faces_vector, self.celeb_label_with_embeddings = self.create_training_data_with_labels(celeb_list,facess,face_embeddings,top_sort=60,best_faces_of_celeb=best_faces_of_celeb)
        return X, labels