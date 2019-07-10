import os
import re
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from sklearn.metrics import confusion_matrix
import shutil

model_dir = 'imagenet'
images_dir = r'.\img_classification'


def create_graph(_model_dir):
    with gfile.FastGFile(os.path.join(_model_dir, "classify_image_graph_def.pb"), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="")


def extract_features(_list_images):
    nb_features = 2048
    features = np.empty((len(_list_images), nb_features))
    labels = []

    create_graph(model_dir)

    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name("pool_3:0")

        for ind, image in enumerate(_list_images):
            if (ind % 100 == 0):
                print("Processing % s..." % (image))
            if not gfile.Exists(image):
                tf.logging.fatal("File dose not exist %s", image)
            image_data = gfile.FastGFile(image, "rb").read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind, :] = np.squeeze(predictions)
            path, file = os.path.split(image)
            labels.append((file.split("_")[1].split(".")[0]))

    return _list_images, features, labels


def svm_classification(_train_dir, _test_dir, _res_dir):
    train_images = [os.path.join(_train_dir, f) for f in os.listdir(_train_dir) if re.search('jpg|JPG|png|PNG', f)]
    test_images = [os.path.join(_test_dir, f) for f in os.listdir(_test_dir) if re.search('jpg|JPG|png|PNG', f)]
    train_images, train_features, train_labels = extract_features(train_images)
    test_images, test_features, test_labels = extract_features(test_images)

    from sklearn.svm import LinearSVC
    clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr')
    clf.fit(train_features, train_labels)
    test_pred = clf.predict(test_features)
    print(confusion_matrix(test_pred, test_labels))

    if not os.path.isdir(_res_dir):
        os.mkdir(_res_dir)
    zip_imgs = zip(test_pred, test_images)

    zip_imgs = list(zip_imgs)

    for img in zip_imgs:
        img_class = img[0]
        src_img_path = img[1]
        _, img_name = os.path.split(src_img_path)
        if not os.path.isdir(os.path.join(_res_dir, img_class)):
            os.mkdir(os.path.join(_res_dir, img_class))
        shutil.copy(src_img_path, os.path.join(_res_dir, img_class, img_name))


def kmeans_classification(_test_dir, _res_dir, n_clusters=5):
    from sklearn.cluster import KMeans
    test_images = [os.path.join(_test_dir, f) for f in os.listdir(_test_dir) if re.search('jpg|JPG|png|PNG', f)]
    test_images, test_features, test_labels = extract_features(test_images)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(test_features)

    if not os.path.isdir(_res_dir):
        os.mkdir(_res_dir)

    zip_imgs = zip(kmeans.labels_, test_images)
    zip_imgs = list(zip_imgs)

    for _img in zip_imgs:
        _img_class = _img[0]
        _src_img_path = _img[1]
        _, _img_name = os.path.split(_src_img_path)
        if not os.path.isdir(os.path.join(_res_dir, str(_img_class))):
            os.mkdir(os.path.join(_res_dir, str(_img_class)))
        shutil.copy(_src_img_path, os.path.join(_res_dir, str(_img_class), _img_name))


def dbscan_classification(_test_dir, _res_dir, eps=50, min_samples=800, n_jobs=1):
    from sklearn.cluster import DBSCAN
    test_images = [os.path.join(_test_dir, f) for f in os.listdir(_test_dir) if re.search('jpg|JPG|png|PNG', f)]
    test_images, test_features, test_labels = extract_features(test_images)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs).fit(test_features)
    zip_imgs = zip(dbscan.labels_, test_images)

    zip_imgs = list(zip_imgs)

    for _img in zip_imgs:
        _img_class = _img[0]
        _src_img_path = _img[1]
        _, _img_name = os.path.split(_src_img_path)
        if not os.path.isdir(os.path.join(_res_dir, str(_img_class))):
            os.mkdir(os.path.join(_res_dir, str(_img_class)))
        shutil.copy(_src_img_path, os.path.join(_res_dir, str(_img_class), _img_name))


if __name__ == '__main__':
    train_dir = r".\img_classification\Train_debug"
    test_dir = r".\img_classification\Test_debug"
    res_dir = r".\img_classification\Result"

    svm_classification(_train_dir=train_dir, _test_dir=test_dir, _res_dir=res_dir)

    kmeans_classification(_test_dir=test_dir, _res_dir=res_dir)

    dbscan_classification(_test_dir=test_dir, _res_dir=res_dir)





