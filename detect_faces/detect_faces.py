# function for face detection with mtcnn
import uuid
from uuid import UUID

import cv2
from PIL import Image
from keras.engine.saving import load_model

from matplotlib import pyplot
from numpy import asarray, load, expand_dims
from mtcnn.mtcnn import MTCNN
import facenet
import tensorflow as tf


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
    # print("ndani")
    for face in faces_array:
        cv2.imwrite(str(face['face_id']) + '.jpg', cv2.cvtColor(face['image_array'], cv2.COLOR_RGB2BGR))
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
    uuu = 0
    for face_array in faces_array:
        uuu+=1
        embedding = get_embedding(model=model, face_pixels=face_array['image_array'])
        face_array['image_embedding'] = embedding
        face_array['face_id'] = str(uuid.uuid4())
        faces_with_embeddings.append(face_array)
    return faces_with_embeddings

def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    import numpy as np
    if len(face_encodings) == 0:
        return np.empty((0))

    #return 1/np.linalg.norm(face_encodings - face_to_compare, axis=1)
    return np.sum(face_encodings*face_to_compare,axis=1)

def _chinese_whispers(encoding_list, threshold=0.78, iterations=20):
    """ Chinese Whispers Algorithm

    Modified from Alex Loveless' implementation,
    http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/

    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold,default 0.6
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate

    Outputs:
        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
            sorted by largest cluster to smallest
    """

    #from face_recognition.api import _face_distance
    from random import shuffle
    import networkx as nx
    from sklearn.preprocessing import Normalizer
    # Create graph
    nodes = []
    edges = []

    # image_paths, encodings = zip(*encoding_list)
    image_paths=list()
    # print(image_paths)
    # exit()
    face_ids = [d.get('face_id', None) for d in encoding_list]
    encodings = [d.get('image_embedding', None) for d in encoding_list]

    in_encoder = Normalizer(norm='l2')
    encodings = in_encoder.transform(encodings)

    # print(len(face_ids))
    # print(encodings)
    # print(len(encodings))
    # exit()

    if len(encodings) <= 1:
        print ("No enough encodings to cluster!")
        return []
    # idx = 0
    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx+1
        print(idx)
        # exit()
        # print(face_encoding_to_check['image_embedding'])
        # exit()
        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': face_ids[idx], 'path': face_ids[idx]})
        nodes.append(node)

        # Facial encodings to compare
        if (idx+1) >= len(encodings):
            # Node is last element, don't create edge
            break

        compare_encodings = encodings[idx+1:]
        # print(compare_encodings)
        # exit()

        distances = face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                # Add edge if facial match
                edge_id = idx+i+2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges = edges + encoding_edges
    # print("tuko ndaaaaaani")
    # exit()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = list(G.nodes())
        shuffle(cluster_nodes)
        cluster_nodes = filter(None,cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}
            neighbors = filter(None, neighbors)

            for ne in neighbors:
                if isinstance(ne, int) and G.node[ne]:
                    # print(G.node[ne])
                    # exit()

                    if G.node[ne]['cluster'] in clusters:
                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']


            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = 0
            #use the max sum of neighbor weights class as current node's class
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.node[node]['cluster'] = max_cluster

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.node.items():

        cluster = data['cluster']
        try:
            path = data['path']
        except KeyError as e:
            path =""

        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)

    # Sort cluster output
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters



def compute_facial_encodings(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
                    embedding_size,nrof_images,nrof_batches,emb_array,batch_size,paths):
    """ Compute Facial Encodings

        Given a set of images, compute the facial encodings of each face detected in the images and
        return them. If no faces, or more than one face found, return nothing for that image.

        Inputs:
            image_paths: a list of image paths

        Outputs:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

    """

    for i in range(nrof_batches):
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = paths
        feed_dict = { images_placeholder:images, phase_train_placeholder:False }
        emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

    facial_encodings = {}
    for x in range(nrof_images):
        facial_encodings[paths[x]] = emb_array[x,:]


    return facial_encodings


def main(face_embeddings, batch_size=20, args=None):
    """ Main

    Given a list of images, save out facial encoding data files and copy
    images into folders of face clusters.

    """
    from os.path import join, basename, exists
    from os import makedirs
    import numpy as np
    import shutil
    import sys

    # if not exists(args.output):
    #     makedirs(args.output)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # image_paths = get_onedir(args.input)
            # image_list, label_list = facenet.get_image_paths_and_labels(train_set)

            # meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            #
            # print('Metagraph file: %s' % meta_file)
            # print('Checkpoint file: %s' % ckpt_file)
            # load_model(args.model_dir, meta_file, ckpt_file)
            load_model('/Users/bildad.namawa/sandbox/jupyter-notebooks/face_recognition/facenet_keras.h5')
            # Get input and output tensors
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            images_placeholder = tf.get_default_graph().get_tensor_by_name("batch_join:0")

            image_size = images_placeholder.get_shape()[1]
            print("image_size:", image_size)
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Runnning forward pass on images')

            nrof_images = len(face_embeddings)
            nrof_batches = int(math.ceil(1.0 * face_embeddings / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            facial_encodings = compute_facial_encodings(sess, images_placeholder, embeddings, phase_train_placeholder,
                                                        image_size,
                                                        embedding_size, nrof_images, nrof_batches, emb_array,
                                                        batch_size, face_embeddings)
            print(face_embeddings)
            exit("kufaaaaaaaa")
            sorted_clusters = cluster_facial_encodings(facial_encodings)
            num_cluster = len(sorted_clusters)

            # Copy image files to cluster folders
            for idx, cluster in enumerate(sorted_clusters):
                # save all the cluster
                cluster_dir = join(args.output, str(idx))
                if not exists(cluster_dir):
                    makedirs(cluster_dir)
                for path in cluster:
                    shutil.copy(path, join(cluster_dir, basename(path)))


def cluster_facial_encodings(facial_encodings):
    """ Cluster facial encodings

        Intended to be an optional switch for different clustering algorithms, as of right now
        only chinese whispers is available.

        Input:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

        Output:
            sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
                sorted by largest cluster to smallest

    """

    if len(facial_encodings) <= 1:
        print("Number of facial encodings must be greater than one, can't cluster")
        return []

    # Only use the chinese whispers algorithm for now
    sorted_clusters = _chinese_whispers(facial_encodings)
    return sorted_clusters


def get_faces_from_directory(folder_path):
    from os import listdir
    from os.path import isfile, join
    images = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    pixels = list()
    for image in images:
        image = folder_path + "/" + image
        pixels = pixels + detect_faces(image)
    # exit()
    return pixels

# load the photo and extract the face
# pixels = detect_faces('/Users/bildad.namawa/sandbox/jupyter-notebooks/bildad4.JPG')
pixels = get_faces_from_directory('/Users/bildad.namawa/sandbox/jupyter-notebooks/input_images')

# exit()
x = create_face_embeddings(pixels)
embeddings = [d.get('image_embedding', None) for d in x]
# save_detected_faces(x)
# encodings = main(embeddings)
sorted_clusters = cluster_facial_encodings(x)
num_cluster = len(sorted_clusters)

for num, cluster in enumerate(sorted_clusters):
    num+=1
    for face in cluster:
        face_array = next((item.get('image_array') for item in x if item["face_id"] == face), False)
        # if face_array. != False:
        cv2.imwrite('./output_images/cluster_'+str(num)+'_'+str(face) + '.jpg', cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR))
# print(sorted_clusters)
print(num_cluster)


