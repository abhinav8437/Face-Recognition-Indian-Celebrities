import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def remove_uncessary_faces_fetched_from_google_using_DBSCAN(face_embeddings,facess,celeb_list):
    db_obj = DBSCAN(eps=0.9,min_samples=5)
    pure_face_embedds = {}
    pure_face_vectors = {}
    for folder in range(len(face_embeddings)):
            predictions = db_obj.fit_predict(np.array(face_embeddings[folder]).reshape(-1,512))
            indexed = pd.Series(predictions).value_counts().index[0]
            pure_face_embedds[celeb_list[folder]] = np.array(face_embeddings[folder]).reshape(-1,512)[predictions==indexed]
            pure_face_vectors[celeb_list[folder]] = np.array(facess[folder])[predictions==indexed]
    return pure_face_embedds, pure_face_vectors