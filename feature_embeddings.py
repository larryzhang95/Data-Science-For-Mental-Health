from __future__ import print_function
import os

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

from scipy.io import wavfile
import pandas as pd
import numpy as np
import six
import tensorflow as tf
import pickle
"""
Feature Embeddings extracted from wavFiles for purpose of cross comparison and building models on top of feature(s)
"""
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
    ppc_batch = np.array(ppc_batch)
    return seq_example,ppc_batch

def getMRMR(wavFile):
    print("TODO: MRMR Implementation/Conversion from MATLAB in progress")

def getGeMaps(wavFile):
    print("TODO: GeMAPS Implementation in progress")
