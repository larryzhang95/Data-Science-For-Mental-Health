from __future__ import print_function
import os

from scipy.io import wavfile
import pandas as pd
import numpy as np
import six
import tensorflow as tf
import pickle
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

#Code:

#Modify these global variables according to your own usage.
pca_param_np = './vggish_pca_params.npz'
vggish_cpkt = './vggish_model.ckpt'
audioDir = './Bipolar_Disorde'

def getAudioSet(wavFile,param_np,tfrFile,cpkt):
    #Input batch
    ex_batch = vggish_input.wavfile_to_examples(wavFile)
    print(ex_batch)

    #Post Processor Initialization to munge model embeddings
    pproc = vggish_postprocess.Postprocessor(param_np)

    #Record Writer to store embeddings
    writer = tf.python_io.TFRecordWriter(tfrFile) if tfrFile is not None else None

    #Model:
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, cpkt)
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        #Inference + Post Processing:
        [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: ex_batch})
        print(embedding_batch)
        ppc_batch = pproc.postprocess(embedding_batch)
        print(ppc_batch)
        seq_example = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
                        tf.train.FeatureList(
                            feature=[
                                tf.train.Feature(
                                    bytes_list=tf.train.BytesList(
                                        value=[embedding.tobytes()]))
                                for embedding in ppc_batch
                            ]
                        )
                }
            )
        )
        print(seq_example)
        if writer:
            writer.write(seq_example.SerializeToString())
            writer.close()
    return seq_example,ppc_batch

def getGeMaps():
    print("GeMaps")


# DATA MANIPULATION ########################################################
def getData(dir,val):
    dirList = os.listdir(dir)
    audioFile = dirList[val]
    return os.path.join(dir,audioFile)

#Get Columns to acquire Bipolar Label(s)
def getBipolarLabels(csv):
    label_df = pd.read_csv(csv)
    label_df = label_df[['Notes','Symptoms']]

    label_df.Symptoms.str.lower()
    label_df['Type'] = label_df.apply(bipolarCase,axis=1)
    label_df = label_df.drop(['Notes'],axis=1)
    return label_df

def bipolarCase(row):
    if "II" in row.Notes:
        return 'BipolarII'
    else:
        return 'BipolarI'
#get usable Y value
def getY(csv):
    df = getBipolarLabels(csv)
    print("Boob")
    df = df.drop(['Type'],axis=1)
    df['label'] = df.apply(labelFunc,axis=1)
    df = df.drop(['Symptoms'], axis=1)
    return df

def labelFunc(row):
    if row.Symptoms == 'neutral':
        return 0
    elif row.Symptoms == 'depressed':
        return 1
    else:
        return 2
def getAllData(dir):
    dataList = os.listdir(dir)
    embeddingsList = []
    for wavFile in dataList:
        audio = os.path.join(dir,wavFile)
        seq, ppc_batch = getAudioSet(audio,pca_param_np,None,vggish_cpkt)
        embeddingsList.append(ppc_batch)
    return embeddingsList
def saveToPickle(vec):
    pickle.dump(vec, open('bipolar_data.p','wb'))
#Main Code
#audio = getData(audioDir,1)
#seq, ppc_batch = getAudioSet(audio,pca_param_np,None,vggish_cpkt)
#label_df = getY('./Bipolar_Data.csv')
#print(label_df.head)
vec=getAllData(audioDir)
saveToPickle(vec)
