# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 12:49:12 2018

@author: Vinodini
"""

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
from helper import *

dataset = "20ng"
#This folder contains weights of the trained model 
folder_name = "1528852501"   
trained_classes =['comp.graphics', 'alt.atheism', 'comp.sys.mac.hardware', 'misc.forsale', 'rec.autos']

dirs = generate_folders(folder_name)
gpu_frac = .2

for tc in trained_classes:
    av_class = [tc]
    if dataset == "amazon":
        datasets1 = data_helpers.get_datasets_localdata("./data/datasets/amazon/train", categories=av_class)
        labels = datasets1['target_names']
        x_net, _ = data_helpers.load_data_labels_amazon(datasets1)
    elif dataset == "20ng":
#         datasets1 = data_helpers.get_datasets_20newsgroup(subset="train", categories=av_class, remove=())
#         labels = datasets1['target_names']
#         x_net, _ = data_helpers.load_data_labels_remove_SW(datasets1)
        datasets1 = data_helpers.get_datasets_localdata("./data/datasets/20newsgroup/train", categories=av_class)
        labels = datasets1['target_names']
        x_net, _ = data_helpers.load_data_labels_amazon(datasets1)
    elif dataset == "reuters":
        datasets1 = data_helpers.get_datasets_localdata("./data/datasets/reuters/train", categories=av_class)
        labels = datasets1['target_names']
        x_net, _ = data_helpers.load_data_labels_amazon(datasets1)
    else:
        print("Invalid dataset!");break;

    path = "./runs/"+folder_name+"/vocab"
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(path)
    x_test = np.array(list(vocab_processor.transform(x_net)))
    checkpoint_path = "./runs/"+folder_name+"/checkpoints/"
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
    graph = tf.Graph()
    with graph.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
        session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False, gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]
            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            batches = data_helpers.batch_iter(list(x_test), 3, 1, shuffle=False) # batch_size

            # Collect the predictions here
            all_predictions = []
            all_probabilities = None
            all_av = []
            for x_test_batch in batches:
                batch_predictions_scores = sess.run([predictions, scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
                single_av = np.array(batch_predictions_scores[1])
                probabilities = softmax(single_av)
                if all_probabilities is not None:
                    all_av = np.concatenate([all_av, single_av])
                    all_probabilities = np.concatenate([all_probabilities, probabilities])
                else:
                    all_av = single_av
                    all_probabilities = probabilities
    all_av = np.array(all_av)
    np.save(dirs['avs_dir']+"/"+tc+".npy", all_av)
    np.save(dirs['mavs_dir']+"/"+tc+".npy", (np.mean(all_av, axis=0)))
    print("Finished calculating AVs for: "+tc+" Shape: "+str(all_av.shape))
print( "All Done.")


k = 10

# K-CLOSEST:
#EUC
#metric_type = "eucos"
#normalized = False
#for c in trained_classes:
#    class_activations = np.load(dirs['avs_dir']+"/"+c+".npy")
#    mean_AV = np.mean(class_activations, axis=0)
#    np.save(dirs['mavs_dir']+"/"+c+".npy", mean_AV)
#    distances_AV_MAV = []
#    for AV in class_activations:
#        temp_dist = distance_metric(AV, mean_AV, metric_type, class_activations, normalized=normalized) #MD
#        distances_AV_MAV.append(temp_dist)
#    closest_AVs = []
#    for i in range(k):
#        closest_AVs.append(class_activations[distances_AV_MAV.index(sorted(distances_AV_MAV)[i])])
#    np.save(dirs['k_closest_dir']+"/"+c+".npy", np.array(closest_AVs)) #EUC
##     np.save(dirs['mahalanobis_dir']+"/"+c+".npy", np.array(closest_AVs)) #MD
#    distances_AV_kClosest = []
#    for AV in class_activations:
#        k_dist_temp = []
#        for cAV in closest_AVs:
#            temp_dist = distance_metric(AV, cAV, metric_type, class_activations, normalized=normalized)
#            k_dist_temp.append(temp_dist)
#        distances_AV_kClosest.append(np.mean(k_dist_temp))
#    distances_AV_kClosest = np.array(distances_AV_kClosest)
#    print (distances_AV_kClosest.shape)
#    np.save(dirs['k_dist_dir']+"/"+c+".npy", distances_AV_kClosest) # EUC
##     np.save(dirs['k_closest_dir2']+"/"+c+".npy", distances_AV_kClosest) # MD
#print( "Done.")

# K - CLOSEST:
tail_size = 30
for c in trained_classes:
    distances = np.load(dirs['k_dist_dir']+"/"+c+".npy")
#     distances = np.load(dirs['k_closest_dir2']+"/"+c+".npy")
    mr = libmr.MR()
    mr.fit_high(np.array(distances), tail_size)
    plt.scatter(distances, 1-mr.w_score_vector(distances))
    pickle.dump(mr.as_binary(), open(dirs['k_dist_wb_dir']+"/"+c+".npy", "wb"))
#     pickle.dump(mr.as_binary(), open(dirs['wb_dir']+"/"+c+".npy", "wb"))
plt.show()
print("Done generating Weibull Model, tailsize: "+str(tail_size))


# K-CLOSEST:
# #MD
metric_type = "md"
normalized = True
for c in trained_classes:
    class_activations = np.load(dirs['avs_dir']+"/"+c+".npy")
    mean_AV = np.mean(class_activations, axis=0)
    np.save(dirs['mavs_dir']+"/"+c+".npy", mean_AV)
    distances_AV_MAV = []
    for AV in class_activations:
        temp_dist = distance_metric(AV, mean_AV, metric_type, class_activations, normalized=normalized) #MD
        distances_AV_MAV.append(temp_dist)
    closest_AVs = []
    for i in range(k):
        closest_AVs.append(class_activations[distances_AV_MAV.index(sorted(distances_AV_MAV)[i])])
#     np.save(dirs['k_closest_dir']+"/"+c+".npy", np.array(closest_AVs)) #EUC
    np.save(dirs['mahalanobis_dir']+"/"+c+".npy", np.array(closest_AVs)) #MD
    distances_AV_kClosest = []
    for AV in class_activations:
        k_dist_temp = []
        for cAV in closest_AVs:
            temp_dist = distance_metric(AV, cAV, metric_type, class_activations, normalized=normalized)
            k_dist_temp.append(temp_dist)
        distances_AV_kClosest.append(np.mean(k_dist_temp))
    distances_AV_kClosest = np.array(distances_AV_kClosest)
    print( distances_AV_kClosest.shape)
#     np.save(dirs['k_dist_dir']+"/"+c+".npy", distances_AV_kClosest) # EUC
    np.save(dirs['k_closest_dir2']+"/"+c+".npy", distances_AV_kClosest) # MD
print("Done.")

# K - CLOSEST:
tail_size = 30
for c in trained_classes:
#     distances = np.load(dirs['k_dist_dir']+"/"+c+".npy")
    distances = np.load(dirs['k_closest_dir2']+"/"+c+".npy")
    mr = libmr.MR()
    mr.fit_high(np.array(distances), tail_size)
    plt.scatter(distances, 1-mr.w_score_vector(distances))
#     pickle.dump(mr.as_binary(), open(dirs['k_dist_wb_dir']+"/"+c+".npy", "wb"))
    pickle.dump(mr.as_binary(), open(dirs['wb_dir']+"/"+c+".npy", "wb"))
plt.show()
print("Done generating Mahalanobis Model, tailsize: "+str(tail_size))




