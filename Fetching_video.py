import cv2
import os
import numpy as np
import pickle as pk
import sys
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
import time
from Paths import celeb_list
from Training_set import Prepare_trainig_set
from sklearn.svm import SVC
from Paths import video_direc,path_to_save_downloaded_images,path_to_save_video_frames

#CONVERTING VIDEO TO IMAGES AND SAVING IN CURRENT DIRECTORY

class Video_to_test_embedds():

    def video_to_images(self):

        vid_cap = cv2.VideoCapture()
        os.chdir(path_to_save_video_frames)
        image_vec = []
        success,image = vid_cap.read()
        count = 0
        while(success==True):
            success,image = vid_cap.read()
            getvalue = vid_cap.get(0)
            if getvalue==20000:
                cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
                image_vec.append(image)
            count += 1

    # TEST IMAGE EMBEDDING!
    # THIS IS ONLY FOR 1 IMAGE AT A TIME
    # TEST IMAGE EMBEDDING!
    # THIS IS ONLY FOR 1 IMAGE AT A TIME
    def test_embeddings(self,test_gray_images):

        face_cascade = cv2.CascadeClassifier(
            '/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
        dict_faces = {}
        test_embedds = []
        test_faces = []
        test_frames = []
        # LOAD FACENT MODEL
        # ==============================================================>
        sys.path.append("/Users/abhinavrohilla")
        from facenet.src import facenet
        model = "/Users/abhinavrohilla/Downloads/VGGFace2 pretrained model"
        sess = tf.Session()
        with sess.as_default():
            facenet.load_model(model)
            image_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # ==============================================================>
            for imga in range(test_gray_images.shape[0]):
                faces_ = face_cascade.detectMultiScale(test_gray_images[imga], 1.3, 5)
                # WORK ON THOSE FRAMES WHICH HAS FACES
                if (len(faces_) != 0):
                    test_frames.append(test_gray_images[imga])
                    faces_in_one_image_embedds = []
                    faces_in_one_image_vector = []
                    test_embedds.append(faces_in_one_image_embedds)
                    test_faces.append(faces_in_one_image_vector)
                    for num, face in enumerate(faces_):
                        dict_faces[num] = face
                    # CREATING A DICT -->> {0:[43,54,133,133],1:[54,76,154,154]}, this is to generalise n number of faces in 1 image
                    for face in dict_faces:
                        for (x, y, w, h) in [dict_faces[face]]:
                            img = cv2.rectangle(test_gray_images[imga], (x, y), (x + w, y + h), (255, 0, 0), 2)
                            bounding_box = img[y:y + h, x:x + w]
                            test_face = cv2.cvtColor(bounding_box, cv2.COLOR_GRAY2BGR)
                            test_face = cv2.resize(test_face, (160, 160))
                            faces_in_one_image_vector.append(test_face)
                            prewhiten_face = facenet.prewhiten(test_face)
                            feed_dict = {image_placeholder: prewhiten_face.reshape(-1, 160, 160, 3),
                                         phase_train_placeholder: False}
                            test_embeddings = sess.run(embeddings, feed_dict=feed_dict)
                            #                 test_embeddings = vector_to_embeddings(test_face.reshape(1,160,160,3))
                            faces_in_one_image_embedds.append(test_embeddings)
        return test_embedds, test_faces, test_frames

    def get_test_embeddings(self):
        # self.video_to_images()
        test_gray_images, test_bgr_images, test_labels = Prepare_trainig_set().image_to_vector("test")
        test_face_embeddings, test_faces, test_frames = self.test_embeddings(test_gray_images)
        return test_face_embeddings, test_faces, test_frames
