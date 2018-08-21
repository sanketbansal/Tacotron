import tensorflow as tf
import pandas as pd
import spacy
import threading
import pandas_datareader.data as pdr
import json,re,random,pickle

from spacy import displacy

class NLP:

    file_schema={"Trainfile":"../data/files/train/","Testfile":"../data/files/test/","dataset":"../data/dataset/","tensorboard":""}
    data={"batch":[]}
    def __init__(self,session):
        self.nlp=spacy.load("en")
        self.session=session
        #self.init=tf.global_variables_initializer()
        #self.session.run(self.init)
        #self.init=tf.initialize_all_variables()
        #self.graph_writer=tf.summary.FileWriter("../data/tensorboard/summary")
        #self.fetch()
        #self.preprocessing()


    def fetch(self):
        with tf.name_scope("fetch"):
            self.fetch_files=tf.WholeFileReader(name="Fetch")
            files=self.match_files(self.file_schema["files"])
            self.fetch_q=tf.train.string_input_producer(files,num_epochs=1,name="FetchQueue")
            self.fetch_data=self.fetch_files.read(self.fetch_q,name="RawData")
        print("fetching done")

    def preprocessing(self):
        with tf.name_scope("PreProcessing"):

            with tf.name_scope("RawData2Sentence"):
                #coord=tf.train.Coordinator()
                #enq_threads=tf.train.start_queue_runners(self.session,coord)
                try:
                    #sents=[]
                    #while True:
                    #self.session.run(tf.local_variables_initializer())
                    #raw_data=self.session.run(self.fetch_data)
                    #sents.extend(self.tokenize(raw_data))
                    data=pd.read_csv(self.file_schema["Trainfile"]+"metadata.csv",sep="|",header=None)
                    sents=data[1].str.strip()
                    sents=sents.values
                    print(sents)
                except Exception as e:
                    #coord.request_stop(e)
                    #coord.join(enq_threads)
                    print(e.message," ","\n")

                finally:
                    #print(sents)
                    length=len(sents)
                    print(length)
                    #sents=tf.convert_to_tensor(sents,dtype=tf.string ,name="SentenceList")
                    sents_dataset=tf.data.Dataset.from_tensor_slices(sents)
                    iterator=sents_dataset.make_initializable_iterator()
                    elem_sent=iterator.get_next()
                    self.session.run(iterator.initializer)
                    #print(self.session.run(element),"\n")

                    self.sents_q=tf.FIFOQueue(length,tf.string,name="SentenceQueue")
                    enq=self.sents_q.enqueue(elem_sent,name="EnqueueSentence")

                    self.qr=tf.train.QueueRunner(self.sents_q,[enq]*78)
                    pass

            with tf.name_scope("Word2vector"):
                coord=tf.train.Coordinator()
                enq_threads=self.qr.create_threads(self.session,coord,start=True)
                #print(enq_threads)
                meta={"type":self.file_schema["dataset"],"name":"TrainDataset","cmd":"","vector":""}
                meta["cmd"]="open"
                self.WriteTFRecords(meta)
                try:
                    vector_data=[]
                    while True:
                        self.session.run(tf.local_variables_initializer())
                        if(not coord.should_stop()):
                            sent=self.session.run(self.sents_q.dequeue(name="DequeueSentence"))
                            #print(sent,"\n")
                            vector_data=[tf.train.Feature(float_list=tf.train.FloatList(value=self.word2vector(token))) for token in self.word_tokenize(sent.decode("utf-8"))]
                            print(len(vector_data))
                            meta["cmd"]="wrt"
                            meta["vector"]=vector_data
                            self.WriteTFRecords(meta)
                        else:
                            break
                except Exception as e:
                    coord.request_stop()
                    coord.join(enq_threads)
                    meta["vector"]=[]
                    print(e.message," ","\n")
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

    def tokenize(self,raw_data):
        with tf.name_scope("Tokenizer"):
            print(raw_data,"\n")
            with tf.name_scope("Regexp"): 
                pass
            try:
                doc=self.nlp(raw_data)
                return [sent.text for sent in list(doc.sents)]
            except Exception as e:
                print(e)

    def word_tokenize(self,sent):
        with tf.name_scope("WordTokenizer"):
            doc=self.nlp(sent)
            return doc

    def word2vector(self,token):
        with tf.name_scope("Word2vector"):
            return token.vector

    def Parse_data(self,serialvector):
        with tf.name_scope("ParseSerializedData"):
            #print(serialvector)
            context_features = {"length": tf.FixedLenFeature([],dtype=tf.int64)}
            sequence_features={"fl":tf.FixedLenSequenceFeature([384],dtype=tf.float32)}

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
