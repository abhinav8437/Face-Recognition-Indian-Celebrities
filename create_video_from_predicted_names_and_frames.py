import cv2
from Model_1 import cosine_or_eucledian_distance


def generate_box_and_name(imagess,texts):
    face_cascade = cv2.CascadeClassifier('/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    for image in range(len(imagess)):
        faces_ = face_cascade.detectMultiScale(imagess[image], 1.3, 5)
        dict_faces = {}
        for num,face in enumerate(faces_):
            dict_faces[num] = face
        for i,face in enumerate(dict_faces):
            for (x,y,w,h) in [dict_faces[face]]:
                img = cv2.rectangle(imagess[image],(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(imagess[image],texts[image][i][0],(x, y), cv2.FONT_HERSHEY_PLAIN, 3.5, (255, 0, 0), 2)

generate_box_and_name(test_frames,final_prediction_knn)
frame_width,frame_height =test_frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('outputttt.avi',fourcc, 20.0, (frame_width,frame_height))
for i in range(len(test_frames)):
        frame = test_frames[i]
        # write the flipped frame
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Release everything if job is finished
out.release()
cv2.destroyAllWindows()