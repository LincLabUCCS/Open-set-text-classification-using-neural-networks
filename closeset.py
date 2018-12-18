#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import yaml
import sys

folder_name = sys.argv[1]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

dataset = "amazon"



#training_classes = ['Lamp', 'Luggage', 'Magazine Subscriptions', 'Mattress', 'Memory Card', 'Microphone', 'Microwave', 'Monitor', 'Mouse', 'Movies TV' ]
#training_classes=['Amplifier', 'Automotive', 'Battery', 'Beauty', 'Cable', 'Camera', 'CDPlayer', 'Clothing', 'Computer', 'Conditioner']
#training_classes =['Fan', 'Flashlight', 'Graphics Card', 'Headphone', 'Home Improvement', 'Jewelry', 'Kindle', 'Kitchen', 'Watch', 'Webcam']
#training_classes =[ 'Patio Lawn Garden', 'Pet Supplies', 'Pillow', 'Printer', 'Projector', 'Rice Cooker', 'Shoes', 'Speaker', 'Subwoofer', 'Table Chair']

training_classes = ['Amplifier', 'Automotive', 'Battery', 'Beauty', 'Cable', 'Camera', 'CDPlayer', 'Clothing', 'Computer', 'Conditioner', 'Fan', 'Flashlight', 'Graphics Card', 'Headphone', 'Home Improvement', 'Jewelry', 'Kindle', 'Kitchen', 'Lamp', 'Luggage']
#training_classes =['Office Products', 'Patio Lawn Garden', 'Pet Supplies', 'Pillow', 'Printer', 'Projector', 'Rice Cooker', 'Shoes', 'Speaker', 'Subwoofer', 'Table Chair', 'Tablet', 'Telephone', 'Tent', 'Toys', 'Video Games', 'Vitamin Supplement', 'Wall Clock', 'Watch', 'Webcam']

#training_classes = [ 'Magazine Subscriptions', 'Mattress', 'Memory Card', 'Microphone', 'Microwave', 'Monitor', 'Mouse', 'Movies TV', 'Musical Instruments', 'Network Adapter','Pillow', 'Printer', 'Projector', 'Rice Cooker', 'Amplifier', 'Automotive', 'Battery', 'Beauty', 'Cable', 'Camera']


#training_classes = ['Amplifier', 'Automotive', 'Battery', 'Beauty', 'Cable', 'Camera', 'CDPlayer', 'Clothing', 'Computer', 'Conditioner', 'Fan', 'Flashlight', 'Graphics Card', 'Headphone', 'Home Improvement', 'Jewelry', 'Kindle', 'Kitchen', 'Lamp', 'Luggage', 'Magazine Subscriptions', 'Mattress', 'Memory Card', 'Microphone', 'Microwave', 'Monitor', 'Mouse', 'Movies TV', 'Musical Instruments', 'Network Adapter', 'Office Products', 'Patio Lawn Garden', 'Pet Supplies', 'Pillow', 'Printer', 'Projector', 'Rice Cooker', 'Shoes', 'Speaker', 'Subwoofer', 'Table Chair', 'Tablet', 'Telephone', 'Tent', 'Toys', 'Video Games', 'Vitamin Supplement', 'Wall Clock', 'Watch', 'Webcam']




#training_classes = ['Watch', 'Graphics Card', 'Shoes','Automotive','Luggage']
#training_classes = ['Graphics Card', 'Shoes']
#training_classes=['Amplifier', 'Automotive', 'Battery', 'Beauty', 'Cable', 'Camera', 'CDPlayer', 'Clothing', 'Computer', 'Conditioner']



#dataset = "20newsgroup"
#training_classes = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
# 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
# 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
# 'talk.politics.misc', 'talk.religion.misc']



#training_classes = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
 #'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
#training_classes = ['sci.electronics', 'sci.med', 'sci.space','comp.graphics', 'talk.religion.misc','comp.windows.x', 'rec.autos']

#training_classes = ['comp.graphics', 'alt.atheism',  'sci.electronics', 'soc.religion.christian','rec.sport.hockey','talk.religion.misc','comp.sys.mac.hardware', 'misc.forsale', 'rec.autos', 'talk.politics.mideast']
#training_classes = ['comp.graphics', 'sci.space','comp.windows.x'  ,'sci.electronics','sci.med','rec.sport.hockey','talk.religion.misc','comp.sys.ibm.pc.hardware', 'rec.autos', 'talk.politics.guns']
#training_classes = [ 'comp.sys.ibm.pc.hardware', 'alt.atheism','talk.politics.guns','sci.med','misc.forsale','sci.crypt','rec.sport.baseball']
#training_classes = ['rec.sport.baseball', 'sci.crypt', 'comp.sys.ibm.pc.hardware', 'sci.med', 'talk.politics.guns' ]
#['comp.graphics', 'alt.atheism', 'comp.sys.mac.hardware', 'misc.forsale', 'rec.autos']
if dataset == "20newsgroup":
    datasets = data_helpers.get_datasets_20newsgroup(subset="test", categories=training_classes, remove=('headers', 'footers', 'quotes'))
    x_raw, y_test = data_helpers.load_data_labels(datasets)
    y_test = np.argmax(y_test, axis=1)

else:

    datasets = data_helpers.get_datasets_localdata("./amazon/test", categories=training_classes) # TODO: tweak parameters in the future
    x_raw, y_test = data_helpers.load_data_labels(datasets) # text is stored in x_test; # labels are stored in y
    y_test = np.argmax(y_test,axis=1)
    print("length of y_test",y_test[:20]) 
print("Total number of test examples: {}".format(len(y_test)))
print(datasets['target_names'])

path = "./runs/"+folder_name+"/vocab"
print "path: "+path
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(path)
x_test = np.array(list(vocab_processor.transform(x_raw)))


# __EVAL__
print("Evaluating...")
checkpoint_path = "./runs/"+folder_name+"/checkpoints/"
checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0] THIS was commented
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), 25, 1, shuffle=False) # batch_size

        # Collect the predictions here
        all_predictions = []
        all_probabilities = None

        for x_test_batch in batches:
            batch_predictions_scores = sess.run([predictions, scores], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
            probabilities = softmax(batch_predictions_scores[1])
            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities, probabilities])
            else:
                all_probabilities = probabilities

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print "Correct: "+str(correct_predictions)
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print(metrics.classification_report(y_test, all_predictions, target_names=datasets['target_names']))
    print(metrics.confusion_matrix(y_test, all_predictions))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw),
                                              [int(prediction) for prediction in all_predictions],
                                              [ "{}".format(probability) for probability in all_probabilities]))

out_path = os.path.join("./runs/"+folder_name+"/checkpoints/", "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
print "Done."
