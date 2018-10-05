import cv2
import os
import numpy as np
import pickle as pk
import sys
import tensorflow as tf
from parse_images_from_google import parse_images_of_celebrities_from_google
from Paths import celeb_list
from Remove_noise import remove_uncessary_faces_fetched_from_google_using_DBSCAN


class Prepare_trainig_set():

    def __init__(self,celeb_list,num_of_pics_for_each_celeb):
        self.celeb_list = celeb_list
        self.num_of_pics_for_each_celeb = num_of_pics_for_each_celeb
        self.celeb_label_with_embeddings = None

    # RETURN IMAGE IN VECTOR FORM
    def image_to_vector(self,dataset="train"):
        bgr_images = []
        gray_scale_images = []
        not_recognized_images_count = 0
        labels = []
        if (dataset == "train"):
            for celeb in self.celeb_list:
                path = os.chdir('/Users/abhinavrohilla/Data/downloads/{}'.format(celeb))
                file_names = os.listdir()
                celeb_folder_gray = []
                celeb_folder_bgr = []
                celeb_label = []
                gray_scale_images.append(celeb_folder_gray)
                bgr_images.append(celeb_folder_bgr)
                labels.append(celeb_label)
                for image_name in range(len(file_names)):
                    path = '/Users/abhinavrohilla/Data/downloads/{}/{}'.format(celeb, file_names[image_name])
                    image = cv2.imread(path)
                    try:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        celeb_folder_gray.append(gray)
                        celeb_label.append(celeb)
                        celeb_folder_bgr.append(image)
                    except:
                        not_recognized_images_count = not_recognized_images_count + 1
                        image_name = image_name + 1

        elif (dataset == "test"):
            path = os.chdir('/Users/abhinavrohilla/video_images')
            total_images_in_folder = os.listdir()
            for image_name in range(len(total_images_in_folder) - 3):
                path = '/Users/abhinavrohilla/video_images/frame{}.jpg'.format(image_name)
                image = cv2.imread(path)

                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray_scale_images.append(gray)
                    bgr_images.append(image)
                except:
                    not_recognized_images_count = not_recognized_images_count + 1
                    image_name = image_name + 1

        return np.array(gray_scale_images), np.array(bgr_images), labels

    # WORKS ONLY FOR GRAY IMAGES
    # THIS IS FOR CREATING TRAINING DATA FACES
    # WORKS ONLY FOR GRAY IMAGES
    # THIS IS FOR CREATING TRAINING DATA FACES
    #GOES IN EVERY FOLDER AND RETURN FACES IN EVERY PICTURE
    #RETURN (len(celeb_list),)
    def detect_face_and_crop(self,gray_images_vec):
        face_cascade = cv2.CascadeClassifier(
            '/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
        test_faces = []
        face_labels = []
        for folder in range(gray_images_vec.shape[0]):
            folder_faces = []
            folder_labels = []
            test_faces.append(folder_faces)
            folder_labels.append(folder_labels)
            for image in range(len(gray_images_vec[folder])):
                dict_faces = {}
                faces_ = face_cascade.detectMultiScale(gray_images_vec[folder][image], 1.3, 5)
                if (len(faces_) != 0):
                    for num, dic_face in enumerate(faces_):
                        dict_faces[num] = dic_face
                    for face in dict_faces:
                        for (x, y, w, h) in [dict_faces[face]]:
                            img = cv2.rectangle(gray_images_vec[folder][image], (x, y), (x + w, y + h), (255, 0, 0), 2)
                            bounding_box = img[y:y + h, x:x + w]
                            try:
                                test_face = cv2.resize(bounding_box, (160, 160))
                                folder_faces.append(test_face)
                                folder_labels.append(self.celeb_list[folder])
                            except:
                                continue
        return np.array(test_faces), face_labels

    def gray_to_bgr_faces(self,facess):
        faces_for_embeddings = []
        for folder in range(len(facess)):
            image_list = []
            faces_for_embeddings.append(image_list)
            for face in range(len(facess[folder])):
                image_list.append(cv2.cvtColor(facess[folder][face], cv2.COLOR_GRAY2BGR))
        return faces_for_embeddings

    # USING FACENET CONVERTING IMAGES VECTOR TO EMBEDDINGS
    # SEND BGR IMAGE
    #RETURN (len(celeb_list),)
    def vector_to_embeddings(self,facess):
        sys.path.append("/Users/abhinavrohilla")
        from facenet.src import facenet
        model = "/Users/abhinavrohilla/Downloads/VGGFace2 pretrained model"
        sess = tf.Session()
        embeddingsss = []
        with sess.as_default():
            facenet.load_model(model)
            image_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            for folder in range(len(facess)):
                folder_list = []
                embeddingsss.append(folder_list)
                for face in range(len(facess[folder])):
                    prewhiten_face = facenet.prewhiten(facess[folder][face])
                    feed_dict = {image_placeholder: prewhiten_face.reshape(-1, 160, 160, 3),
                                 phase_train_placeholder: False}
                    embedds = sess.run(embeddings, feed_dict=feed_dict)
                    folder_list.append(embedds)
        return embeddingsss

    def create_training_data_with_labels(self,celeb_list, celeb_label_with_embeddings, pure_face_vectors):
        labels = []
        X = []
        vectors = []
        for i in range(len(celeb_label_with_embeddings)):
            for j in range(len(celeb_label_with_embeddings[celeb_list[i]])):
                labels.append(celeb_list[i])
                X.append(celeb_label_with_embeddings[celeb_list[i]][j])
                vectors.append(pure_face_vectors[celeb_list[i]][j])
        X = np.array(X)
        return X, labels, np.array(vectors)

    def get_training_set_with_labels(self):
        # parse_images_of_celebrities_from_google(celebrities=celeb_list,num_of_images=self.num_of_pics_for_each_celeb)
        # gray_images, bgr_images, labels = self.image_to_vector("train")
        # facess, labels = self.detect_face_and_crop(gray_images)
        # pk.dump(facess, open("facess.p", "wb"))
        # os.chdir("/Users/abhinavrohilla/Data/downloads/Salman Khan/")
        facess = pk.load(open("facess.p", "rb"))
        # FACENET REQUIRES BGR IMAGES
        # bgr_faces_for_embeddings = self.gray_to_bgr_faces(facess=facess)
        # face_embeddings = self.vector_to_embeddings(bgr_faces_for_embeddings)
        # pk.dump(face_embeddings,open("face_embeddings.p","wb"))
        face_embeddings = pk.load(open("face_embeddings.p","rb"))
        pure_face_embedds, pure_face_vectors = remove_uncessary_faces_fetched_from_google_using_DBSCAN(face_embeddings,facess,self.celeb_list)
        X, labels, X_vectors = self.create_training_data_with_labels(self.celeb_list, pure_face_embedds, pure_face_vectors)
        return X, labels, pure_face_embedds