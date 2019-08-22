# function for face detection with mtcnn
import cv2
from PIL import Image
from keras.engine.saving import load_model

from matplotlib import pyplot
from numpy import asarray, load, expand_dims
from mtcnn.mtcnn import MTCNN


def detect_faces(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face

    detected_faces = list()
    for result in results:
        print(result)
        # exit()
        x1, y1, width, height = result['box']
        # use absolute pixels
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        # resize image
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_image_array = asarray(image)
        detected_face = {
            "image_array": face_image_array,
            "image_metadata": result
        }
        detected_faces.append(detected_face)
    return detected_faces


def show_detected_face(face_array):
    pyplot.subplot(1, 1, 1)
    pyplot.axis('off')
    pyplot.imshow(face_array)
    pyplot.show()
    return


def save_detected_faces(faces_array):
    i = 0
    for face in faces_array:
        cv2.imwrite('bildad5e43_' + str(i) + '.jpg', cv2.cvtColor(face['image_array'], cv2.COLOR_RGB2BGR))
        i = i + 1


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]



def create_face_embeddings(faces_array):
    model = load_model('/Users/bildad.namawa/sandbox/jupyter-notebooks/face_recognition/facenet_keras.h5')
    faces_with_embeddings = list()
    for face_array in faces_array:
        embedding = get_embedding(model=model, face_pixels=face_array['image_array'])
        face_array['image_embedding'] = embedding


# load the photo and extract the face
pixels = detect_faces('/Users/bildad.namawa/sandbox/jupyter-notebooks/bildad5.JPG')
save_detected_faces(pixels)
# create_face_embeddings(pixels)