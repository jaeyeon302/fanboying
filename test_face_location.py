import face_recognition
import cv2
import time
TEST_IMG = "jeon.jpg"

def test_the_result_of_face_location(img_file_name):
    img = face_recognition.load_image_file(img_file_name)
    img = cv2.resize(img,dsize=(0,0),fx=0.5,fy=1.0,interpolation=cv2.INTER_AREA)
    s = time.time()
    face_locations = face_recognition.face_locations(img )
    e = time.time()
    print("time: {} s".format(e-s))
    print(face_locations)

if __name__ == "__main__":
    test_the_result_of_face_location(TEST_IMG)