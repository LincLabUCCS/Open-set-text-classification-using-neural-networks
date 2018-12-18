import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
import libmr
import scipy.spatial.distance as ssd
import pickle
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))
def format20n_input(trained_classes, untrained_classes):
    datasets1 = data_helpers.get_datasets_20newsgroup(subset="test", categories=trained_classes, remove=('headers', 'footers', 'quotes'))
    x_raw1, y_raw1 = data_helpers.load_data_labels(datasets1)
    y_test1 = np.argmax(y_raw1, axis=1)
    labels1 = datasets1['target_names']
    dataset2 = data_helpers.get_datasets_20newsgroup(subset="test", categories=untrained_classes, remove=('headers', 'footers', 'quotes'))
    x_raw2, y_test2 = data_helpers.load_data_labels(dataset2)
    y_test2 = np.add(np.argmax(y_test2, axis=1), len(trained_classes))
    labels2 = dataset2['target_names']
    x_net = x_raw1 + x_raw2
    y_net = np.append(y_test1, y_test2)
    labels = labels1 + labels2
    return (x_net, y_net, labels)
def format_amazon_input(trained_classes, untrained_classes):
    datasets1 = data_helpers.get_datasets_localdata("./data/datasets/amazon/test", categories=trained_classes)
    labels1 = datasets1['target_names']
    x_raw1, y_raw1 = data_helpers.load_data_labels_amazon(datasets1)
    y_test1 = np.argmax(y_raw1, axis=1)
    dataset2 = data_helpers.get_datasets_localdata("./data/datasets/amazon/test", categories=untrained_classes)
    x_raw2, y_test2 = data_helpers.load_data_labels_amazon(dataset2)
    y_test2 = np.add(np.argmax(y_test2, axis=1), len(trained_classes))
    labels2 = dataset2['target_names']
    x_net = x_raw1 + x_raw2
    y_net = np.append(y_test1, y_test2)
    labels = labels1 + labels2
    return (x_net, y_net, labels)

def format20n_input_local(trained_classes, untrained_classes):
    datasets1 = data_helpers.get_datasets_localdata("./data/datasets/20newsgroup/test", categories=trained_classes)
    labels1 = datasets1['target_names']
    x_raw1, y_raw1 = data_helpers.load_data_labels_amazon(datasets1)
    y_test1 = np.argmax(y_raw1, axis=1)
    dataset2 = data_helpers.get_datasets_localdata("./data/datasets/20newsgroup/test", categories=untrained_classes)
    x_raw2, y_test2 = data_helpers.load_data_labels_amazon(dataset2)
    y_test2 = np.add(np.argmax(y_test2, axis=1), len(trained_classes))
    labels2 = dataset2['target_names']
    x_net = x_raw1 + x_raw2
    y_net = np.append(y_test1, y_test2)
    labels = labels1 + labels2
    return (x_net, y_net, labels)

def format_reuters_input_local(trained_classes, untrained_classes):
    datasets1 = data_helpers.get_datasets_localdata("./data/datasets/reuters/test", categories=trained_classes)
    labels1 = datasets1['target_names']
    x_raw1, y_raw1 = data_helpers.load_data_labels_amazon(datasets1)
    y_test1 = np.argmax(y_raw1, axis=1)
    dataset2 = data_helpers.get_datasets_localdata("./data/datasets/reuters/test", categories=untrained_classes)
    x_raw2, y_test2 = data_helpers.load_data_labels_amazon(dataset2)
    y_test2 = np.add(np.argmax(y_test2, axis=1), len(trained_classes))
    labels2 = dataset2['target_names']
    x_net = x_raw1 + x_raw2
    y_net = np.append(y_test1, y_test2)
    labels = labels1 + labels2
    return (x_net, y_net, labels)


def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector/norm if norm != 0 else vector
def mark_unknowns(y_test, trained_classes):
    y_new = []
    for e in y_test:
        if e > (len(trained_classes) - 1):
            y_new.append('u')
        else:
            y_new.append(e)
    return y_new
#def eucos(a, b):
#    return ssd.euclidean(a,b)/200. + ssd.cosine(a,b)
def euclidean(a, b):
    return ssd.euclidean(a,b)
def cosine(a, b):
    return ssd.cosine(a,b)
def md(u, v, avs):
    return ssd.mahalanobis(u, v, VI=np.linalg.inv(np.cov(avs, rowvar=False)))
def h_mean(a, b):
    return float(2*a*b)/(a+b)
def macro_metrics(M):
    precisions = []
    recalls = []
    fscores = []
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if i == j:
                tp = M[i][j]
        recalls.append(float(tp)/(np.sum(M[i], axis=0)))
        precisions.append(float(tp)/(np.sum(M, axis=0)[i]))
    pavg = np.mean(precisions)
    ravg = np.mean(recalls)
    for i in range(len(precisions)):
        try:
            class_fscore = float(2*precisions[i]*recalls[i])/float(precisions[i]+recalls[i])
        except:
            class_fscore = 0
        fscores.append(class_fscore)
    fm = np.mean(fscores)
    return [pavg, ravg, fm]
def calc_metrics(y_test, new_pred, trained_classes):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    trained_class_indices = [x for x in range(len(trained_classes))]
    for i, elem in enumerate(y_test):
        if elem == new_pred[i] and elem != 18 and new_pred[i] != 18:
            tp+=1  # TP
        elif elem==18 and new_pred[i] == 18:
            tn+=1 #TN
#             tp+=1 #TN
        elif elem in trained_class_indices and new_pred[i] == 18:
            fn+=1 # FN
        elif elem == 18 and new_pred[i] in trained_class_indices:
            fp+=1 #FP  
        elif elem in trained_class_indices and new_pred[i] in trained_class_indices and new_pred[i] != elem:
            fp+=1
        else:
            print(i)
    total_correct = tp+tn
    accuracy = total_correct/float(len(y_test))
    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    f1 = 2*(precision*recall)/(precision + recall)
    return [accuracy, precision, recall, f1]

def distance_metric(a, b, dist_type, cov_inp, normalized=True):
    if normalized is True:
        a = normalize(a)
        b = normalize(b)
    if dist_type == "md":
        return md(a, b, cov_inp)
    elif dist_type == "eu":
        return euclidean(a, b)
    elif dist_type == "cos":
        return cosine(a, b)
#    elif dist_type == "eucos":
#        return eucos(a, b)
    else:
        print ("Error, Invalid distance metric.")
        return None

def generate_folders(folder_name):
    avs_dir = "./data/avs/avs_"+folder_name
    mavs_dir = "./data/avs/avs_"+folder_name+"/mavs"
    dist_dir = "./data/avs/avs_"+folder_name+"/distances_mahalanobis"
    wb_dir = "./data/avs/avs_"+folder_name+"/wb_models"
    k_closest_dir = "./data/avs/avs_"+folder_name+"/k_closest"
    k_closest_dir2 = "./data/avs/avs_"+folder_name+"/k_closest2"
    k_dist_dir = "./data/avs/avs_"+folder_name+"/k_distances"
    k_dist_wb_dir = "./data/avs/avs_"+folder_name+"/k_dist_wb_models"
    k_medoids_dir = "./data/avs/avs_"+folder_name+"/k_medoids_dir"
    if not os.path.exists(avs_dir):
        os.makedirs(avs_dir)
    if not os.path.exists(mavs_dir):
        os.makedirs(mavs_dir)
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    if not os.path.exists(wb_dir):
        os.makedirs(wb_dir)
    if not os.path.exists(k_dist_dir):
        os.makedirs(k_dist_dir)
    if not os.path.exists(k_closest_dir):
        os.makedirs(k_closest_dir)
    if not os.path.exists(k_closest_dir2):
        os.makedirs(k_closest_dir2)
    if not os.path.exists(k_dist_wb_dir):
        os.makedirs(k_dist_wb_dir)
    if not os.path.exists(k_medoids_dir):
        os.makedirs(k_medoids_dir)
    return {'avs_dir': avs_dir, 'mavs_dir': mavs_dir, 'mahalanobis_dir': dist_dir, 'wb_dir': wb_dir, 'k_closest_dir': k_closest_dir, 'k_dist_dir': k_dist_dir, 'k_dist_wb_dir': k_dist_wb_dir, 'k_medoids_dir':k_medoids_dir, 'k_closest_dir2':k_closest_dir2}