import tensorflow as tf
from tensorflow.contrib import ffmpeg

import pandas as pd
import numpy as np
import librosa

from sklearn import preprocessing

import threading,os
import json,re,random,pickle,time


class AP:

    file_schema={"Trainfile":"../data/files/train/audio","Testfile":"../data/files/test/","dataset":"../data/dataset/","tensorboard":""}
    data={"batch":[]}
 
    def __init__(self,session):
        self.session=session
        #self.fetch()
        #self.preprocessing()


    def fetch(self):
        with tf.name_scope("fetch"):
            files=self.match_files(self.file_schema["Trainfile"])
            self.fetch_q=tf.train.string_input_producer(files,num_epochs=1,name="FetchQueue")
        print("fetching done")


    def preprocessing(self):
        with tf.name_scope("PreProcessing"):
            with tf.name_scope("ReadRawData"):
                coord=tf.train.Coordinator()
                enq_threads=tf.train.start_queue_runners(self.session,coord)
                print("Enqueued Threads:",enq_threads,"\n")
                self.mel_spectrogram_data=[]
                meta={"type":self.file_schema["dataset"],"name":"TrainAudio","cmd":"","vector":""}
                meta["cmd"]="open"
                self.WriteTFRecords(meta)
                try:
                    while True:
                        self.session.run(tf.local_variables_initializer())
                        file=self.session.run(self.fetch_q.dequeue(name="DequeFiles"))
                        #print(file)
                        waveform,sample_rate =  librosa.load(file)
                        #librosa.display.waveplot(waveform, sr=sampling_rate)

                        frame = tf.contrib.signal.frame(waveform, frame_length=128, frame_step=20)
                        magnitude_spectrogram = tf.abs(tf.contrib.signal.stft(waveform ,frame_length=256, frame_step=64, fft_length=256))

                        #log_offset = 1e-6
                        #log_magnitude_spectrogram = tf.log(magnitude_spectrogram + log_offset)

                        num_spectrogram_bins = magnitude_spectrogram.shape[-1].value
                        
                        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 64
                        
                        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
                        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
                        upper_edge_hertz)
                        
                        mel_spectrogram = tf.tensordot(
                        magnitude_spectrogram, linear_to_mel_weight_matrix, 1)
                        
                        mel_spectrogram.set_shape(magnitude_spectrogram.shape[:-1].concatenate(
                        linear_to_mel_weight_matrix.shape[-1:]))

                        log_offset = 1e-6
                        log_mel_spectrogram = tf.log(mel_spectrogram + log_offset)

                        #self.mel_spectrogram_data.append(log_mel_spectrogram)

                        '''num_mfccs = 13
                        # Keep the first `num_mfccs` MFCCs.
                        mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
                            log_mel_spectrogram)'''

                        log_mel_spectrogram=self.session.run(log_mel_spectrogram)
                        print(len(log_mel_spectrogram[-1])," ",len(log_mel_spectrogram[-3]))

                        vector_data=[tf.train.Feature(float_list=tf.train.FloatList(value=frame)) for frame in log_mel_spectrogram ]
                        print(len(vector_data))
                        meta["cmd"]="wrt"
                        meta["vector"]=vector_data
                        self.WriteTFRecords(meta)

                except Exception as e:
                    coord.request_stop(e)
                    coord.join(enq_threads)
                    print(e,"\n")
                finally:
                    meta["vector"]=[]
                    meta["cmd"]="close"
                    self.WriteTFRecords(meta)
                    pass

            '''with tf.name_scope("ProcessRawData"):

                self.img_q=tf.FIFOQueue(length,tf.float32,name="IMGQueue")
                self.lab_q=tf.FIFOQueue(length,tf.float32,name="LABQueue")

                self.lab=tf.data.Dataset().from_tensor_slices()
                #self.img=self.lab.prefetch(100)
                print(self.lab)

                img_iterator=self.img.make_initializable_iterator()
                img=img_iterator.get_next()
                lab_iterator=self.lab.make_initializable_iterator()
                lab=lab_iterator.get_next()
                #self.session.run(iterator.initializer)
                #print(self.session.run(element),'\n')

                enq_img=self.img_q.enqueue(img,name="EnqueueIMGData")
                enq_lab=self.lab_q.enqueue(lab,name="EnqueueLABData")
                self.img_qr=tf.train.QueueRunner(self.img_q,[enq_img]*128)
                self.lab_qr=tf.train.QueueRunner(self.lab_q,[enq_lab]*128)
                pass'''

            '''with tf.name_scope("SaveTFRecords"):
                try:
                    coord=tf.train.Coordinator()
                    self.session.run(img_iterator.initializer)
                    self.session.run(lab_iterator.initializer)

                    img_enq_threads=self.img_qr.create_threads(self.session,coord,start=True)
                    lab_enq_threads=self.lab_qr.create_threads(self.session,coord,start=True)
                    #print(img_enq_threads)
                    #print(lab_enq_threads)

                    meta={"type":self.file_schema["dataset"],"name":"TrainDataset","cmd":"","vector":""}
                    meta["cmd"]="open"
                    self.WriteTFRecords(meta)
                    while True:
                        self.session.run(tf.local_variables_initializer())
                        if(not coord.should_stop()):
                            img,lab=self.session.run([self.img_q.dequeue(name="DequeueIMGData"),self.lab_q.dequeue(name="DequeueLABData")])
                            #print(len(img)," ",lab,"\n")
                            lab=tf.one_hot(lab,10)
                            #print(self.session.run(lab))
                            lab = self.session.run(lab)
                            vector_data=[tf.train.Feature(float_list=tf.train.FloatList(value=img)),tf.train.Feature(float_list=tf.train.FloatList(value=lab))]
                            meta["cmd"]="wrt"
                            meta["vector"]=vector_data
                            self.WriteTFRecords(meta)                          
                        else:
                            break
                except Exception as e:
                    coord.request_stop(e)
                    coord.join(img_enq_threads)
                    #coord.join(lab_enq_threads)
                    meta["vector"]=[]
                    print(e.message,"\n")
                    pass
                finally:
                    meta["cmd"]="close"
                    self.WriteTFRecords(meta)
                    pass'''

        print("preprocessing done")


    def match_files(self,type):
        with tf.name_scope("Match"):
            files=tf.matching_files(type+"*",name="match")
            #files=tf.train.match_filenames_once(type,name="match")
            #print(files)
            return files            


    def Parse_data(self,serialvector):
        with tf.name_scope("ParseSerializedData"):
            #print(serialvector)
            context_features = {"length": tf.FixedLenFeature([],dtype=tf.int64)}
            sequence_features={"fl":tf.FixedLenSequenceFeature([64],dtype=tf.float32)}

            _,sequence=tf.parse_single_sequence_example(serialized=serialvector,context_features=context_features,sequence_features=sequence_features)
            print(sequence["fl"])
            return sequence["fl"]

    def WriteTFRecords(self,meta):
        with tf.name_scope("WTFrecord"):
            if(meta["cmd"]=="open"):
                self.writer=tf.python_io.TFRecordWriter(meta["type"]+meta["name"]+".tfrecords")
            elif(meta["cmd"]=="wrt"):
                feature_list={}
                length=len(meta["vector"])
                
                feature_list["fl"]=tf.train.FeatureList(feature=meta["vector"])
                record=tf.train.FeatureLists(feature_list=feature_list)
                example=tf.train.SequenceExample(feature_lists=record)

                example.context.feature["length"].int64_list.value.append(length)
                self.writer.write(example.SerializeToString())
            elif(meta["cmd"]=="close"):    
                self.writer.close()                               

