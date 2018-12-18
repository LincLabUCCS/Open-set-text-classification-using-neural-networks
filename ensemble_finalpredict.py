# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 09:10:26 2018

@author: Vinodini
"""

import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
from sklearn import metrics
import scipy.spatial.distance as ssd
import sys
import pickle
import libmr
import random
from PyNomaly import loop
from helper import *
from sklearn.ensemble import IsolationForest

def inp_data(a, b):
    c = list(a)
    c.append(b)
    return np.array(c)

openset_test = True


dataset ="20ng"
folder_name = "1544055461"
trained_classes = ['comp.windows.x', 'misc.forsale', 'talk.politics.guns', 'sci.electronics', 'rec.autos']
#dataset = "amazon"
#folder_name = "1540581200"
#trained_classes = ['Shoes', 'Battery', 'Pillow', 'Graphics Card','Rice Cooker']


dirs = generate_folders(folder_name)
untrained_class_size = 5 #trained on 5 classes and testing on 10 classes where 5 classes are unseen


if dataset == "amazon":
    all_classes = ['Amplifier', 'Automotive', 'Battery', 'Beauty', 'Cable', 'Camera', 'CDPlayer', 'Clothing', 'Computer', 'Conditioner', 'Fan', 'Flashlight', 'Graphics Card', 'Headphone', 'Home Improvement', 'Jewelry', 'Kindle', 'Kitchen', 'Lamp', 'Luggage', 'Magazine Subscriptions', 'Mattress', 'Memory Card', 'Microphone', 'Microwave', 'Monitor', 'Mouse', 'Movies TV', 'Musical Instruments', 'Network Adapter', 'Office Products', 'Patio Lawn Garden', 'Pet Supplies', 'Pillow', 'Printer', 'Projector', 'Rice Cooker', 'Shoes', 'Speaker', 'Subwoofer', 'Table Chair', 'Tablet', 'Telephone', 'Tent', 'Toys', 'Video Games', 'Vitamin Supplement', 'Wall Clock', 'Watch', 'Webcam']
elif dataset == "20ng":
    all_classes = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
elif dataset == "reuters":
    all_classes = ['acq', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade']
else:
    print("Invalid dataset!")

untrained_classes = random.sample(list(set(all_classes) - set(trained_classes)), untrained_class_size)
testing_classes = trained_classes + untrained_classes
print("Testing classes", testing_classes)
print("length of testing classes" ,len(testing_classes))

if openset_test == True:
    if dataset == "amazon":
        x_net, y_net, labels = format_amazon_input(trained_classes, untrained_classes)
    elif dataset == "20ng":
        x_net, y_net, labels = format20n_input_local(trained_classes, untrained_classes)
    elif dataset == "reuters":
        x_net, y_net, labels = format_reuters_input_local(trained_classes, untrained_classes)
else:
    if dataset == "amazon":
        dataset = data_helpers.get_datasets_localdata("./data/datasets/amazon/test", categories=trained_classes)
        labels = dataset['target_names']
        x_net, y_raw1 = data_helpers.load_data_labels_amazon(dataset)
        y_net = np.argmax(y_raw1, axis=1)
    elif dataset == "20ng":
        pass
        # code this

path = "./runs/"+folder_name+"/vocab"
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(path)
x_test = np.array(list(vocab_processor.transform(x_net)))
print("After transformation")
print(x_test.shape)
print(x_test)
print( "Done.")

gpu_frac = 0.2

# __EVAL__
print("Evaluating...")

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
        print( "Batching...")
        batches = data_helpers.batch_iter(list(x_test), 5, 1, shuffle=False) # batch_size
        # Collect the predictions here
        all_predictions = []
        all_probabilities = None
        all_av = []
        print ("Batching Done. Evaluation Begin.")
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
print( "Done.")

# create a list with trained classes and u for unknown classes, y_net = [1 2 0 ...5 8 7] is modified to y_act = [1 2 0 ...5 u u] as only 5 classes are trained and 8, 7 are unknown. 
# We put a high value 120 for 'u' else leave y_act as it is. so y_act = [ 1 0 0 ....120 120 120]
y_act = mark_unknowns(y_net, trained_classes);y_act = [120 if x=='u' else x for x in y_act];
print(y_act[:20])

#Local Outlier Factor for novelty dtection by setting novelty to true.
print("\n Local outlier Factor")
from sklearn.neighbors import LocalOutlierFactor
all_unnormalized_scores1 = []
trained_class_avs1 = {}
lofs = {}
thresholds_lof = {}
for tc in trained_classes:
	trained_class_avs1[tc] = np.load("./data/avs/avs_"+folder_name+"/"+tc+".npy")
	lof_clf = LocalOutlierFactor(n_neighbors=60, novelty = True, contamination = 0.2)
	lofs[tc] = lof_clf.fit(trained_class_avs1[tc])
print("Done loading AVs")
first2pairs = {k:lofs[k] for k in list(lofs)[:2]}
#print(first2pairs)

final_lof_score =[]
final_predict_scores = []

for av in all_av:
    predict_scores = []
    un_nm_scores1 = []
    lof_score_samples = []

    for tc in labels[:len(trained_classes)]:
        clf = lofs[tc]
        predict_scores.append(clf.predict(av.reshape(1,-1))[0])
        un_nm_scores1.append(clf.decision_function(av.reshape(1,-1))[0])
        lof_score_samples.append(clf.score_samples(av.reshape(1,-1))[0])

    all_unnormalized_scores1.append(un_nm_scores1)
    final_lof_score.append(lof_score_samples)
    final_predict_scores.append(predict_scores)
print("length of decision function",len(all_unnormalized_scores1))  # negative values are outliers and positive values are inliners
print("length of final_lof_score", len(final_lof_score))
print("predict_scores", final_predict_scores[:20])
#print("\n decision function scores",all_unnormalized_scores1[:80])
#print("\n LOf score samples")
#print(final_lof_score[:10])


lof_pred = []
for i, value in enumerate(final_predict_scores):           #visit again to put another loop instead of hard coding
	if final_predict_scores[i][0] == -1 and final_predict_scores[i][1] == -1 and final_predict_scores[i][2] == -1 and final_predict_scores[i][3] == -1 and final_predict_scores[i][4] == -1: 
		lof_pred.append(120)
	elif final_predict_scores[i][0] == 1:
		lof_pred.append(0)
	elif final_predict_scores[i][1] == 1:
		lof_pred.append(1)
	elif final_predict_scores[i][2] == 1:
                lof_pred.append(2)
        elif final_predict_scores[i][3] == 1:
                lof_pred.append(3)
 	elif final_predict_scores[i][4] == 1:
                lof_pred.append(4)
        
print("length of lof_pred", len(lof_pred))
print(lof_pred[:20])

# Isolation Forest Algorithm 
print("\n Isolation forest")
all_unnormalized_scores = []

trained_class_avs = {}
isolation_forests = {}
thresholds = {}
for tc in trained_classes:
    trained_class_avs[tc] = np.load("./data/avs/avs_"+folder_name+"/"+tc+".npy")
    clf = IsolationForest(random_state=42)
    isolation_forests[tc] = clf.fit(trained_class_avs[tc])
    thresholds[tc] = clf.threshold_
#print ("Done loading AVs")

##isloation_forest dictionary contains each class classifier clf. Key is trained class name and value is clf object of isolation forest
first2pairs = {k:isolation_forests[k] for k in list(isolation_forests)[:2]}
#print(first2pairs);
#threshold values for 2 classes [-0.0317 -0.0132]
#print (thresholds.values())

for av in all_av:
    un_nm_scores = []
    
    for tc in labels[:len(trained_classes)]:
        clf = isolation_forests[tc]
        un_nm_scores.append(clf.decision_function(av.reshape(1,-1))[0])
	
    all_unnormalized_scores.append(un_nm_scores)

#add thresholds
all_unnormalized_scores.append(thresholds.values())
from sklearn.preprocessing import minmax_scale
f = minmax_scale(np.array(all_unnormalized_scores))
scores = f[:-1]
thres = f[-1]
iso_pred = []

print("scores", scores[:10])
for v in scores:
#     temp = []
    thres_max = {}
    for i, s in enumerate(v):
        if s > thres[i]:
            thres_max[i] = s - thres[i]
    if len(thres_max) == 0:
        iso_pred.append(120)
    else:
        iso_pred.append(max(thres_max, key=thres_max.get))
        
loop_pred = iso_pred        
print(len(loop_pred))

print(loop_pred[:20])

      
# kNN averaged tried adding additional outlier detectors but did not perform well
# EUCOS
# metric_type = "eucos"
# normalized = False
# threshold = 0.4

# eucos_pred = []
# max_csps = []
# max_osps = []
# os_probs = []
# weibull_models = {}
# trained_class_avs = {}
# closest_avs = {}
# tempvar = len(y_net)
# print (tempvar)
# for tc in trained_classes:
#     weibull_models[tc] = libmr.load_from_binary(pickle.load(open("./data/avs/avs_"+folder_name+"/k_dist_wb_models/"+tc+".npy", "rb")))
#     trained_class_avs[tc] = np.load("./data/avs/avs_"+folder_name+"/"+tc+".npy")
#     closest_avs[tc] = np.load("./data/avs/avs_"+folder_name+"/k_closest/"+tc+".npy")
# print ("Done loading Weibull models.")
# sri = 0
# for av in tqdm(all_av):
# #     if sri > 15:break;
#     d = {}
#     p = {}
#     for tc in trained_classes:
#         k_dist_temp = []
#         for e in closest_avs[tc]:
#             cov_inp = trained_class_avs[tc]
#             temp_dist = distance_metric(av, e, metric_type, cov_inp, normalized=normalized)
#             k_dist_temp.append(temp_dist)
#         d[tc] = np.mean(k_dist_temp)
#         p[tc] = 1-weibull_models[tc].w_score_vector(np.array([d[tc]], dtype="double"))
#     final_p = {} 
#     for i, v in enumerate(p.values()):
#         if v[0] > threshold:
#             final_p[i] = v[0]
#     if len(final_p) == 0:
#         eucos_pred.append(120)
#     else:
#         eucos_pred.append(int(max(final_p, key=final_p.get)))
#     sri+=1
# print ("Done.")

#y_pred = eucos_pred
#metric_type = "EUCOS"
#normalized = False
#correct_predictions = float(sum(np.array(y_pred) == np.array(y_act)))
#print ("Model: "+folder_name+"  metric: "+metric_type+"  normalized: "+str(normalized))
#print( "Correct: "+str(correct_predictions))
#print("Total number of test examples: {}".format(len(y_act)))
#print ("Accuracy: "+str(metrics.accuracy_score(y_act, y_pred)))
#k = metrics.precision_recall_fscore_support(y_act, y_pred, average='macro')
#print("F1-Score: {:g}".format(k[2]))
#print( k)
#M = metrics.confusion_matrix(y_act, y_pred)
#print(metrics.classification_report(y_act, y_pred, target_names=trained_classes+['Unknown']))
#print( M)



# kNN averaged
# MAHALANOBIS model 
metric_type = "md"
normalized = True
threshold = 0.8

md_pred = []
max_csps = []
max_osps = []
os_probs = []
weibull_models = {}
trained_class_avs = {}
closest_avs = {}
tempvar = len(y_net)
print (tempvar)
for tc in trained_classes:
    weibull_models[tc] = libmr.load_from_binary(pickle.load(open("./data/avs/avs_"+folder_name+"/wb_models/"+tc+".npy", "rb")))
    trained_class_avs[tc] = np.load("./data/avs/avs_"+folder_name+"/"+tc+".npy")
    closest_avs[tc] = np.load("./data/avs/avs_"+folder_name+"/distances_mahalanobis/"+tc+".npy")
print ("Done loading Weibull models.")
for av in tqdm(all_av):
    d = {}
    p = {}
    for tc in trained_classes:
        k_dist_temp = []
        for e in closest_avs[tc]:
            cov_inp = trained_class_avs[tc]
            temp_dist = distance_metric(av, e, metric_type, cov_inp, normalized=normalized)
            k_dist_temp.append(temp_dist)
        d[tc] = np.mean(k_dist_temp)
        p[tc] = 1-weibull_models[tc].w_score_vector(np.array([d[tc]], dtype="double"))
    av_os_prob = list(p.values())
    total_csp = np.sum(av_os_prob, axis=0)
    osp = 1-total_csp
    max_csp = p[max(p, key=p.get)][0]
    if float(max_csp) < float(osp) or float(max_csp) < threshold:
        md_pred.append(120)
    else:
        md_pred.append(labels.index(str(max(p, key=p.get))))
print ("Done.")


# print("--------------------------------------------------------\n")
# print("actual",y_act[-20:])
# print("lof",lof_pred[-20:])
# print("loop",loop_pred[-20:])
# print("md",md_pred[-20:])
# print("--------------------------------------------------------\n")



ensemble_pred = []
size = len(y_act)
for i in range(size):
    loop = loop_pred[i];md = md_pred[i]; lof = lof_pred[i]                              
    if loop == 120 and md == 120 and lof == 120:
        ensemble_pred.append(120)
    elif loop == md and loop != lof:
        ensemble_pred.append(loop)    
    elif md == lof and md != loop:
        ensemble_pred.append(md)
    elif lof == loop and lof != md:
        ensemble_pred.append(lof)
    elif lof != md and md != loop and lof != loop:
        ensemble_pred.append(md) # md given highest priority tie breaker
    elif eucos == md and md == loop:
        ensemble_pred.append(md)
    else:
        print(i)
        
        
        
#ensemble model prediction        
y_pred = ensemble_pred
print(labels)
print(dataset)
metric_type = "ENSEMBLE"
correct_predictions = sum(np.array(y_pred) == np.array(y_act))
print("Model: "+folder_name+"  metric: "+metric_type)
prin ("Correct: "+str(correct_predictions))
print("Total number of test examples: {}".format(len(y_act)))
print ("Accuracy: "+str(metrics.accuracy_score(y_act, y_pred)))
k = metrics.precision_recall_fscore_support(y_act, y_pred, average='macro')
print("F1-Score: {:g}".format(k[2]))
print(k)
M = metrics.confusion_matrix(y_act, y_pred)
print(metrics.classification_report(y_act, y_pred, target_names=trained_classes+['Unknown']))
print(M)


# invidividual model prediction "Isolation Forest"
y_pred = loop_pred
metric_type = "Isolation Forest"
normalized = False
correct_predictions = float(sum(np.array(y_pred) == np.array(y_act)))
print( "Model: "+folder_name+"  metric: "+metric_type+"  normalized: "+str(normalized))
print( "Correct: "+str(correct_predictions))
print("Total number of test examples: {}".format(len(y_act)))
print ("Accuracy: "+str(metrics.accuracy_score(y_act, y_pred)))
k = metrics.precision_recall_fscore_support(y_act, y_pred, average='macro')
print("F1-Score: {:g}".format(k[2]))
print(k)      
M = metrics.confusion_matrix(y_act, y_pred)
print(metrics.classification_report(y_act, y_pred, target_names=trained_classes+['Unknown']))
print (M)

# invidividual model prediction Mahalanobis model
y_pred = md_pred
metric_type = "MD"
normalized = True
correct_predictions = float(sum(np.array(y_pred) == np.array(y_act)))
print ("Model: "+folder_name+"  metric: "+metric_type+"  normalized: "+str(normalized))
print ("Correct: "+str(correct_predictions))
print("Total number of test examples: {}".format(len(y_act)))
print ("Accuracy: "+str(metrics.accuracy_score(y_act, y_pred)))
k = metrics.precision_recall_fscore_support(y_act, y_pred, average='macro')
print("F1-Score: {:g}".format(k[2]))
print (k)
M = metrics.confusion_matrix(y_act, y_pred)
print(metrics.classification_report(y_act, y_pred, target_names=trained_classes+['Unknown']))
print( M)

# invidividual model prediction LOF model
y_pred = lof_pred
metric_tyep = "LOF"
normalized = False
correct_predictions = float(sum(np.array(y_pred) == np.array(y_act)))
print ("Model: "+folder_name+"  metric: "+metric_type+"  normalized: "+str(normalized))
print ("Correct: "+str(correct_predictions))
print("Total number of test examples: {}".format(len(y_act)))
print( "Accuracy: "+str(metrics.accuracy_score(y_act, y_pred)))
k = metrics.precision_recall_fscore_support(y_act, y_pred, average='macro')
print("F1-Score: {:g}".format(k[2]))
print (k)
M = metrics.confusion_matrix(y_act, y_pred)
print(metrics.classification_report(y_act, y_pred, target_names=trained_classes+['Unknown']))
print (M)



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

