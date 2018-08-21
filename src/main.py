from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing

import threading,os
import json,re,random,pickle,time

import txtproc as txt_proc
#import imgproc as img_proc 
import audioproc as audio_proc

import feed_forward_network as fmodel
#import LSTM as smodel
import CNN as Conv

class MLP:

    file_schema={"Trainfile":"../data/files/train/","Testfile":"../data/files/test/","dataset":"../data/dataset/","tensorboard":""}
    data={"batch":[]}

    inputwidth=384
    classes=64
    variable_list=[]
    summaries={"scalar":[]}

    def __init__(self):
        #tf.enable_eager_execution()
        with tf.name_scope("DataInput"):
            self.DataSrcfeatures=tf.placeholder(tf.float32,[None,self.inputwidth],name="SrcFeatures")
            self.DataTrgfeatures=tf.placeholder(tf.float32,[None,self.classes],name="TargetFeatures")

        self.session=tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True))

        self.graph_writer=tf.summary.FileWriter("../data/tensorboard/summary/graph/")
        self.train_writer=tf.summary.FileWriter("../data/tensorboard/summary/train/")
        self.test_writer=tf.summary.FileWriter("../data/tensorboard/summary/test/")

    

        self.txt_proc=txt_proc.NLP(self.session)
        self.audio_proc=audio_proc.AP(self.session)

        self.train()

        #Testing Data Not Available
        #self.test()

    def modelArch(self):
        with tf.name_scope("Encoder"):

            with tf.name_scope("FeatureConvolution"):
                input_layer = tf.reshape(self.DataSrcfeatures, [-1, 1, self.inputwidth,1])

                conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=128,
                kernel_size=[1,5],
                padding="same",
                activation=tf.nn.relu)

                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1,2], strides=2)

                conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=32,
                    kernel_size=[1,5],
                    padding="same",
                    activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1,2], strides=1)

                conv3 = tf.layers.conv2d(
                    inputs=pool2,
                    filters=64,
                    kernel_size=[1,5],
                    padding="same",
                    activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[1,2], strides=1)

                self.conv_flatten = tf.contrib.layers.flatten(pool3,scope="ConvFlatO")

            with tf.name_scope("BidirectionalLSTM"):

                no_hlayer=1
                num_units=[256]

                cells_fw = [tf.nn.rnn_cell.BasicLSTMCell(unit) for unit in num_units]
                cells_bw = [tf.nn.rnn_cell.BasicLSTMCell(unit) for unit in num_units]

                '''num_units=256
                encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
                #encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_cell)

                flatten_shape = self.conv_flatten.shape[-1]
                inputs = tf.reshape(self.conv_flatten,[1,-1,flatten_shape])
                print(inputs)'''


                flatten_shape = self.conv_flatten.shape[-1]
                inputs = tf.reshape(self.conv_flatten,[1,-1,flatten_shape])

                self.bi_outputs = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw,
                cells_bw,
                inputs,
                initial_states_fw=None,
                initial_states_bw=None,
                dtype=tf.float32,
                sequence_length=None,
                parallel_iterations=None,
                time_major=False,
                scope=None
                )
                self.encoder_outputs = tf.concat(self.bi_outputs[0],-1)
                self.encoder_state = []
                for index,units in enumerate(num_units):
                    self.encoder_state.append(self.bi_outputs[1][index])
                    self.encoder_state.append(self.bi_outputs[2][index])
                self.encoder_state = self.encoder_state[-1]

                '''self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, inputs, dtype=tf.float32)'''
                pass

        with tf.name_scope("Decoder"):

            with tf.name_scope("Helper"):

                input = self.DataTrgfeatures

                #print(input.shape[-1])

                self.helper = tf.contrib.seq2seq.TrainingHelper(
                input, [1607])

                print(self.helper)
                pass

            with tf.name_scope("AttentionMechanism"):

                #self.encoder_outputs = tf.transpose(self.encoder_outputs, [1, 0, 2])
                
                num_units=256
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=self.encoder_outputs,
                memory_sequence_length=[29])

                '''cell = tf.contrib.rnn.GRUCell(num_units=num_units)'''
                
                no_units=[256,256]
                cell = [tf.contrib.rnn.GRUCell(num_units=units) for units in no_units] 
                cell = tf.nn.rnn_cell.MultiRNNCell(cell)

                self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell, attention_mechanism, attention_layer_size=num_units / 4)
                pass

            with tf.name_scope("Decoder"):

                '''num_units = [256]
                decoder_cell = [tf.nn.rnn_cell.BasicLSTMCell(units) for units in num_units]
                decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cell)'''

                inputs = tf.reshape(self.DataTrgfeatures,[1,-1,64])

                decoder_initial_state = self.decoder_cell.zero_state([1],tf.float32).clone(
                cell_state=tuple(self.encoder_state))

                self.decoder_outputs, _ = tf.nn.dynamic_rnn(
                self.decoder_cell, inputs,initial_state=decoder_initial_state, dtype=tf.float32)

                self.decoder_outputs = tf.reshape(self.decoder_outputs,[-1,64])
                self.decoder_outputs = tf.layers.dense(inputs=self.decoder_outputs,units=64)


 
                '''num_units=256
                decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)


                decoder_initial_state = self.decoder_cell.zero_state([1],tf.float32).clone(
                cell_state=tuple(self.encoder_state))


                #inputs = tf.reshape(self.DataTrgfeatures,[-1,1,64])
                input = tf.map_fn(lambda x: x,self.session.run(self.trg_iterator.get_next()))                
                self.decoder_outputs, _ =  tf.contrib.legacy_seq2seq.rnn_decoder(
                    input,
                    decoder_initial_state,
                    self.decoder_cell,
                    loop_function=None,
                    scope=None
                )'''


                '''projection_layer = tf.layers.Dense(
                64, use_bias=False)

                decoder_initial_state = self.decoder_cell.zero_state([1],tf.float32).clone(
                cell_state=tuple(self.encoder_state)[-1])

                decoder = tf.contrib.seq2seq.BasicDecoder(
                self.decoder_cell, self.helper, decoder_initial_state, output_layer=projection_layer)

                print(decoder)

                self.decoder_outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=False,
                impute_finished=True,
                maximum_iterations=20)'''

                '''self.logits = self.decoder_outputs.rnn_output'''

        with tf.name_scope("PostNet"):
            decoder_shape = self.decoder_outputs.shape[-1]
            
            input_layer = tf.reshape(self.decoder_outputs, [-1, 1, decoder_shape,1])

            conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=128,
            kernel_size=[1,5],
            padding="same",
            activation=tf.tanh)

            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1,2], strides=2)

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=32,
                kernel_size=[1,5],
                padding="same",
                activation=tf.tanh)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1,2], strides=1)

            conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=64,
                kernel_size=[5,5],
                padding="same",
                activation=tf.nn.relu)
            self.pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[1,2], strides=2)

            self.postnet_flatten = tf.contrib.layers.flatten(self.pool3,scope="ConvFlatO")

        with tf.name_scope("FinalOutput"):

            dense = tf.layers.dense(inputs = self.postnet_flatten, units = 1024, activation = tf.nn.relu)
            dropout = tf.layers.dropout(
             inputs=dense, rate=0.4)

            # Logits Layer
            self.logits = tf.layers.dense(inputs=dropout, units=64)

            #self.logits = tf.reshape(self.logits,[,64])

            self.decoder_outputs = tf.reshape(self.decoder_outputs,[-1,64])

            self.mel_output = self.logits + self.decoder_outputs
            pass

    def Optimize(self):
        with tf.name_scope("Optimization_cross_entropy"):
            batch_size = 1

            '''crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.DataTrgfeatures, logits=self.decoder_outputs)'''

            crossent = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.DataTrgfeatures,self.mel_output),name="cross_entropy")

            '''train_loss = (tf.reduce_sum(crossent * target_weights)/
            batch_size)'''

            params = tf.trainable_variables()
            gradients = tf.gradients(crossent, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 3)

            learning_rate = .003
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

            self.summaries["scalar"].append({"scope":"Optimization_cross_entropy","name":"cross_entropy","value":crossent})

    def train(self):
        with tf.name_scope("Training"):
            with tf.name_scope("ReadDataset"):
                src_files=self.match_files(self.file_schema["dataset"]+"TrainDataset")
                self.src_dataset=tf.data.TFRecordDataset(src_files)

                trg_files=self.match_files(self.file_schema["dataset"]+"TrainAudio")
                self.trg_dataset=tf.data.TFRecordDataset(trg_files)
                pass

            with tf.name_scope("TrainModels"):
                coord=tf.train.Coordinator()

                self.src_dataset = self.src_dataset.map(self.txt_proc.Parse_data,num_parallel_calls=2) # Parse the record into tensors.
                #self.src_dataset = self.src_dataset.repeat(1) # Repeat the input indefinitely.
                #self.src_dataset = self.src_dataset.batch(1)
                #self.src_dataset = self.src_dataset.prefetch(18)
                self.src_iterator = self.src_dataset.make_initializable_iterator()
                self.session.run(self.src_iterator.initializer)

                #self.DataSrcfeatures=self.session.run(self.src_iterator.get_next())
                #print(self.DataSrcfeatures,"\n")

                self.trg_dataset = self.trg_dataset.map(self.audio_proc.Parse_data,num_parallel_calls=2) # Parse the record into tensors.
                #self.trg_dataset = self.trg_dataset.repeat(1) # Repeat the input indefinitely.
                #self.trg_dataset = self.trg_dataset.batch(1)
                #self.trg_dataset = self.trg_dataset.prefetch(18)
                self.trg_iterator = self.trg_dataset.make_initializable_iterator()
                self.session.run(self.trg_iterator.initializer)

                #self.DataTrgfeatures=self.session.run(self.trg_iterator.get_next())
                #print(self.DataTrgfeatures,"\n")

                try:
                    self.modelArch()
                    self.Optimize()
                    self.summarize()
                    self.graph_writer.add_graph(self.session.graph)

                    iter=0
                    while(iter<=9):
                        src_features = self.session.run(self.src_iterator.get_next())
                        trg_features = self.session.run(self.trg_iterator.get_next())
                        print(len(trg_features))

                        self.session.run(tf.global_variables_initializer())
                        output=self.session.run([self.update_step,self.merged],feed_dict={self.DataSrcfeatures:src_features,self.DataTrgfeatures:trg_features})

                        #print(output[0].shape)

                        self.train_writer.add_summary(output[-1],iter)

                        print("ITERATION: ", iter," ","\n")
                        if(iter%2==0):
                            self.saver.save(self.session,"../data/tensorboard/variable_tensor/NMT.ckpt",global_step=iter)
                        iter=iter+1
                    pass

                except Exception as e:
                    coord.request_stop(e)
                    print("Main Trainig Error: ", e,"\n")
                    pass
                else:
                    coord.request_stop()
                    #coord.join(enq_threads)
                    pass
        print("Training done")

    def test(self):
        with tf.name_scope("Testing"):
            with tf.name_scope("ReadDataset"):
                files=self.match_files(self.file_schema["dataset"]+"TrainDataset")
                self.src_dataset=tf.data.TFRecordDataset(files)
                pass

            with tf.name_scope("TestModel"):
                coord=tf.train.Coordinator()
                self.src_dataset = self.src_dataset.map(self.Parse_data) # Parse the record into tensors.
                #self.src_dataset = self.src_dataset.repeat(1) # Repeat the input indefinitely.
                self.src_dataset = self.src_dataset.batch(1)
                self.iterator=self.src_dataset.make_initializable_iterator()
                #self.session.run(self.iterator.initializer)    

                try:
                    self.neural1=model.neuralNet(self.session)
                    
                    self.neural1.input_features=self.neural1.Datafeatures
                    self.neural1.input_labels=self.neural1.Datalabels    

                    self.neural1.iterator=self.iterator
                    self.neural1.Test_model("model")

                except Exception as e:
                    coord.request_stop(e)
                    print(e.message,"\n")
                else:
                    coord.request_stop()
                    #coord.join(enq_threads)
        print("Testing done")

    def match_files(self,type):
        with tf.name_scope("Match"):
            files=tf.matching_files(type+"*",name="match")
            #files=tf.train.match_filenames_once(type,name="match")
            #print(files)
            return files      

    def summarize(self):
        for typ,summaries in self.summaries.items():
            for summary in summaries:
                with tf.name_scope(summary["scope"]):
                    method="tf.summary."+typ+"(a,b)"
                    eval(method,{"tf":tf},{"a":summary["name"],"b":summary["value"]})
        self.merged=tf.summary.merge_all()
        #print(self.variable_list)
        self.saver=tf.train.Saver()
      

init=MLP()