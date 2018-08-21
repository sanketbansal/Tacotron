import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing

import threading,os
import json,re,random,pickle,time
from mnist import MNIST


class IP:

    file_schema={"Trainfile":"../data/files/train/","Testfile":"../data/files/test/","dataset":"../data/dataset/","tensorboard":""}
    data={"batch":[]}
 
    def __init__(self,session):
        self.session=session
        self.fetch()
        self.preprocessing()


    def fetch(self):
        with tf.name_scope("fetch"):
            files=self.match_files(self.file_schema["Trainfile"])
            self.fetch_q=tf.train.string_input_producer(files,num_epochs=1,name="FetchQueue")
        print("fetching done")


    def preprocessing(self):
        with tf.name_scope("PreProcessing"):
            with tf.name_scope("ReadRawData"):
                coord=tf.train.Coordinator()
                #enq_threads=tf.train.start_queue_runners(self.session,coord)
                #print(enq_threads)
                try:
                    #while True:
                    #self.session.run(tf.local_variables_initializer())
                    #file=self.session.run(self.fetch_q.dequeue(name="DequeFiles"))
                    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
                    self.train_images = mnist.train.images # Returns np.array
                    self.train_images = tf.cast(self.train_images,dtype=tf.float32)
                    self.train_images = tf.Session().run(self.train_images)
                    self.train_labels = np.asarray(mnist.train.labels, dtype=np.float32)
                    self.test_images = mnist.test.images
                    self.test_labels = np.asarray(mnist.train.labels, dtype=np.int32)
                    #print(self.train_labels)  
                    if(coord.should_stop()):
                        pass
                except Exception as e:
                    coord.request_stop(e)
                    print(e.message," ","80","\n")
                else:
                    coord.request_stop()
                    #coord.join(enq_threads)
                    pass

            with tf.name_scope("ProcessRawData"):
                try:
                    '''os.system("rscript analysis.r")'''
                    length=len(self.train_images)
                    print(length)
                except Exception as e:
                    print(e)
                else:
                    self.img_q=tf.FIFOQueue(length,tf.float32,name="IMGQueue")
                    self.lab_q=tf.FIFOQueue(length,tf.float32,name="LABQueue")

                    self.img=tf.data.Dataset().from_tensor_slices(self.train_images[2800:])
                    #self.img=self.img.prefetch(100)
                    print(self.img)
                    self.lab=tf.data.Dataset().from_tensor_slices(self.train_labels[2800:])
                    #self.img=self.lab.prefetch(100)
                    print(self.lab)

                    img_iterator=self.img.make_initializable_iterator()
                    img=img_iterator.get_next()
                    lab_iterator=self.lab.make_initializable_iterator()
                    lab=lab_iterator.get_next()
                    #self.session.run(iterator.initializer)
                    #print(self.session.run(element),"\n")

                    enq_img=self.img_q.enqueue(img,name="EnqueueIMGData")
                    enq_lab=self.lab_q.enqueue(lab,name="EnqueueLABData")
                    self.img_qr=tf.train.QueueRunner(self.img_q,[enq_img]*128)
                    self.lab_qr=tf.train.QueueRunner(self.lab_q,[enq_lab]*128)
                    pass

            with tf.name_scope("SaveTFRecords"):
                try:
                    coord=tf.train.Coordinator()
                    self.session.run(img_iterator.initializer)
                    self.session.run(lab_iterator.initializer)

                    img_enq_threads=self.img_qr.create_threads(self.session,coord,start=True)
                    lab_enq_threads=self.lab_qr.create_threads(self.session,coord,start=True)
                    #print(img_enq_threads)
                    #print(lab_enq_threads)
                    #vector_data=[]

                    meta={"type":self.file_schema["dataset"],"name":"TrainDataset2","cmd":"","vector":""}
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
                    pass
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
                    pass

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
            sequence_features={"featu":tf.FixedLenFeature([784],dtype=tf.float32),"val":tf.FixedLenFeature([10],dtype=tf.float32)}
            sequence=tf.parse_single_example(serialized=serialvector,features=sequence_features)
            #sequence=tf.sparse_tensor_to_dense(sequence["fl"])
            print(sequence)
            return sequence

    def WriteTFRecords(self,meta):
        with tf.name_scope("WTFrecord"):
            if(meta["cmd"]=="open"):
                self.writer=tf.python_io.TFRecordWriter(meta["type"]+meta["name"]+".tfrecords")
            elif(meta["cmd"]=="wrt"):
                features={}
                #for index,vector in enumerate(vectors):
                length=len(meta["vector"])
                features["featu"]=meta["vector"][0]
                features["val"]=meta["vector"][1]
                record=tf.train.Features(feature=features)
                example=tf.train.SequenceExample(context=record)

                example.context.feature["length"].int64_list.value.append(length)
                self.writer.write(example.SerializeToString())
            elif(meta["cmd"]=="close"):    
                self.writer.close()                               

