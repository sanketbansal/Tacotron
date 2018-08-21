import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import numpy.matlib

class neuralNet:

	layers=[]
	layer_schema={"neurons":"", "weights":{"ig":" ","og":" ","fg":" ","ct":" "},"biases":{"ig":" ","og":" ","fg":" ","ct":" "},"feedback":{"ig":" ","og":" ","fg":" ","ct":" "}}
	inputwidth=784
	classes=10
	no_hlayer=2
	neurons_list=[inputwidth,[50,26]]
	variable_list={}
	summaries={"histogram":[],"scalar":[]}
	pre_ct=0
	Total_neurons=0
	pre_feeds={}
	testing_set=[]
	training_set=[]
	input_features=0
	input_labels=0
	iterator=None
	viterator=None #Iterator for validation
	epoch=1
	layerOutputs={}
	layerOutputsVal={}
	Total_loss=[]
	loss=0
	losses=[]

	def __init__(self,session):
		with tf.name_scope("Datainput"):
			self.Datafeatures=tf.placeholder(tf.float32,name="features")
			self.Datalabels=tf.placeholder(tf.float32,name="labels")
		self.session=session
		
		#self.session=tf_debug.LocalCLIDebugWrapperSession(self.session)
		#self.session.add_tensor_filter("nan_filter",tf_debug.has_inf_or_nan)

		self.init=tf.global_variables_initializer()
		self.graph_writer=tf.summary.FileWriter("../data/tensorboard/summary/graph/")
		self.train_writer=tf.summary.FileWriter("../data/tensorboard/summary/train")
		self.test_writer=tf.summary.FileWriter("../data/tensorboard/summary/test")
		self.session.run(self.init)

	@classmethod
	def Layersconf(cls):
		#cls.session=tf.Session()
		cls.Total_neurons=cls.classes+cls.neurons_list[0]
		for i in cls.neurons_list[1]:
			cls.Total_neurons=cls.Total_neurons+i
		for type in ["ig","fg","og","ct_"]:
			cls.pre_feeds[type]=tf.zeros([1,cls.Total_neurons],name="pre_feeds"+type)

		cls.Total_loss=tf.zeros([1,neuralNet.classes],name="Total_loss")
		cls.Inputlayer(cls.neurons_list[0])
		cls.Hiddenlayers(cls.neurons_list[1])
		cls.Outputlayer()
		print("conf Done")
		print(cls.layers)

	@classmethod
	def Inputlayer(cls,neurons):
		cls.layers.append({"neurons":neurons})
		#cls.pre_ct.append(tf.zeros([1,cls.layers[0]["neurons"]]))
		cls.layers[0]["weights"]={}
		cls.layers[0]["biases"]={}
		cls.layers[0]["feedback"]={}
		for type in ["ig","fg","og","ct_"]:
			with tf.name_scope("Inputlayer"+type):
				cls.layers[0]["weights"][type] = tf.Variable(tf.random_normal([cls.inputwidth,cls.layers[0]["neurons"]]),trainable=
				True,dtype=tf.float32,name="weights")
				cls.variable_list["weights"+type+str(0)]=cls.layers[0]["weights"][type]

				cls.layers[0]["biases"][type] = tf.Variable(tf.zeros([1,cls.layers[0]["neurons"]]),trainable=True,dtype=tf.float32,name="biases")
				cls.variable_list["biases"+type+str(0)]=cls.layers[0]["biases"][type]

				cls.layers[0]["feedback"][type] = tf.Variable(tf.random_normal([cls.Total_neurons,cls.layers[0]["neurons"]]),trainable=True,dtype=tf.float32,name="feedback")
				cls.variable_list["feedback"+type+str(0)]=cls.layers[0]["feedback"][type]

				cls.summaries["histogram"].append({"scope":"Inputlayer"+type,"name":"weights","value":cls.layers[0]["weights"][type]})
				cls.summaries["histogram"].append({"scope":"Inputlayer"+type,"name":"biases","value":cls.layers[0]["biases"][type]})
				cls.summaries["histogram"].append({"scope":"Inputlayer"+type,"name":"feedback","value":cls.layers[0]["feedback"][type]})

			cls.layer_no_out=cls.layers[0]["neurons"]
		print(cls.layers[0],"\n")

	@classmethod
	def Hiddenlayers(cls,neurons_list):
		cls.layers.append([])
		for index , neurons in enumerate(neurons_list):
			cls.layers[1].append({"neurons":neurons})
			#cls.pre_ct.append(tf.zeros([1,cls.layers[1][index]["neurons"]]))
			cls.layers[1][index]["weights"]={}
			cls.layers[1][index]["biases"]={}
			cls.layers[1][index]["feedback"]={}
			for type in ["ig","fg","og","ct_"]:
				with tf.name_scope("Hiddenlayer"+type+str(index)):
					cls.layers[1][index]["weights"][type] = tf.Variable(tf.random_normal([cls.layer_no_out,cls.layers[1][index]["neurons"]]),trainable=True,dtype=tf.float32,name="weights")
					cls.variable_list["weights"+type+str(1)+str(index)]=cls.layers[1][index]["weights"][type]

					cls.layers[1][index]["biases"][type] = tf.Variable(tf.ones([1,cls.layers[1][index]["neurons"]]),trainable=True,dtype=tf.float32,name="biases")
					cls.variable_list["biases"+type+str(1)+str(index)]=cls.layers[1][index]["biases"][type]

					cls.layers[1][index]["feedback"][type] = tf.Variable(tf.random_normal([cls.Total_neurons,cls.layers[1][index]["neurons"]]),trainable=True,dtype=tf.float32,name="feedback")
					cls.variable_list["feedback"+type+str(1)+str(index)]=cls.layers[1][index]["feedback"][type]

					cls.summaries["histogram"].append({"scope":"Hiddenlayer"+type+str(index),"name":"weights","value":cls.layers[1][index]["weights"][type]})
					cls.summaries["histogram"].append({"scope":"Hiddenlayer"+type+str(index),"name":"biases","value":cls.layers[1][index]["biases"][type]})
					cls.summaries["histogram"].append({"scope":"Hiddenlayer"+type+str(index),"name":"feedback","value":cls.layers[1][index]["feedback"][type]})

			cls.layer_no_out=cls.layers[1][index]["neurons"]
		print(cls.layers[1],"\n")

	@classmethod
	def Outputlayer(cls):
		cls.layers.append({"neurons":cls.classes})
		cls.layers[2]["weights"]={}
		cls.layers[2]["biases"]={}
		cls.layers[2]["feedback"]={}
		cls.pre_ct=tf.zeros([1,cls.classes])
		print(cls.pre_ct)
		for type in ["ig","fg","og","ct_"]:
			with tf.name_scope("Outputlayer"+type):
				cls.layers[2]["weights"][type] = tf.Variable(tf.random_normal([cls.layer_no_out,cls.classes]),trainable=True,dtype=tf.float32,name="weights")
				cls.variable_list["weights"+type+str(2)]=cls.layers[2]["weights"][type]

				cls.layers[2]["biases"][type] = tf.Variable(tf.ones([1,cls.classes]),trainable=True,dtype=tf.float32,name="biases")
				cls.variable_list["biases"+type+str(2)]=cls.layers[2]["biases"][type]

				cls.layers[2]["feedback"][type] = tf.Variable(tf.random_normal([cls.Total_neurons,cls.layers[2]["neurons"]]),trainable=True,dtype=tf.float32,name="feedback")
				cls.variable_list["feedback"+type+str(2)]=cls.layers[2]["feedback"][type]

				cls.summaries["histogram"].append({"scope":"Outputlayer"+type,"name":"weights","value":cls.layers[2]["weights"][type]})
				cls.summaries["histogram"].append({"scope":"Outputlayer"+type,"name":"biases","value":cls.layers[2]["biases"][type]})
				cls.summaries["histogram"].append({'scope': "Outputlayer"+type, 'name': "feedback", 'value': cls.layers[2]["feedback"][type]})
		print(cls.layers[2],"\n")

	def update_feeds(self,type):
		feeds=[]
		for feed in self.layerOutputsVal[type]:
			#print(feed)
			feeds.extend(feed[-1])
		self.layerOutputsVal[type]=[]
		self.pre_feeds[type]=feeds
		self.pre_feeds[type]=tf.reshape(self.pre_feeds[type],[1,neuralNet.Total_neurons])
		print(self.pre_feeds[type],"\n")

	def Activation(self):
		self.layerOutputs={}
		self.layerI={}
		self.layerO={}
		for type in ["ig","fg","og","ct_"]:
			self.layerOutputs[type]=[]
			with tf.name_scope("Inputlayer"+type):
				self.layerI[type]=tf.add_n([tf.matmul(self.input_features,neuralNet.layers[0]["weights"][type]),tf.matmul(self.pre_feeds[type],neuralNet.layers[0]["feedback"][type]),neuralNet.layers[0]["biases"][type]],name="Weightedinput")
				if(type=="ct_"):
					self.layerO[type]=tf.tanh(self.layerI[type],name="Outputtanh")
				else:
					self.layerO[type]=tf.nn.softmax(self.layerI[type],name="Outputsoftmax")
				self.layerOutputs[type].append(self.layerO[type])

				neuralNet.summaries["histogram"].append({"scope":"Inputlayer"+type,"name":"Weightedinput","value":self.layerI[type]})
				neuralNet.summaries["histogram"].append({"scope":"Inputlayer"+type,"name":"Outputrelu","value":self.layerO[type]})
		print("InputLayer Done","\n")

		for index, layer in enumerate(neuralNet.layers[1]):
			for type in ["ig","fg","og","ct_"]:
				with tf.name_scope("Hiddenlayer"+type+str(index)):
					self.layerI[type]=tf.add_n([tf.matmul(self.layerO[type],layer["weights"][type]),tf.matmul(self.pre_feeds[type],layer["feedback"][type]),layer["biases"][type]],name="Weightedinput")
					if(type=="ct_"):
						self.layerO[type]=tf.tanh(self.layerI[type],name="Outputtanh")
					else:
						self.layerO[type]=tf.nn.softmax(self.layerI[type],name="Outputsoftmax")
					self.layerOutputs[type].append(self.layerO[type])

					neuralNet.summaries["histogram"].append({"scope":"Hiddenlayer"+type+str(index),"name":"Weightedinput","value":self.layerI[type]})
					neuralNet.summaries["histogram"].append({"scope":"Hiddenlayer"+type+str(index),"name":"Outputrelu","value":self.layerO[type]})
		print("HiddenLayersDone","\n")

		for type in ["ig","fg","og","ct_"]:
			with tf.name_scope("Outputlayer"+type):
				self.layerI[type]=tf.add_n([tf.matmul(self.layerO[type],neuralNet.layers[2]["weights"][type]),tf.matmul(self.pre_feeds[type],neuralNet.layers[2]["feedback"][type]),neuralNet.layers[2]["biases"][type]],name="Weightedinput")
				if(type=="ct_"):
					self.layerO[type]=tf.tanh(self.layerI[type],name="Outputtanh")
				else:
					self.layerO[type]=tf.nn.softmax(self.layerI[type],name="Outputsoftmax")
				self.layerOutputs[type].append(self.layerO[type])

				neuralNet.summaries["histogram"].append({"scope":"Outputlayer"+type,"name":"Weightedinput","value":self.layerI[type]})
				neuralNet.summaries["histogram"].append({"scope":"Outputlayer"+type,"name":"Outputsoftmax","value":self.layerO[type]})
		print("OutputLayersDone","\n")

		#print(tf.multiply(self.layerO["fg"],self.pre_ct))
		with tf.name_scope("post_output"):
			self.layerO["ct"]=tf.add(tf.multiply(self.layerO["fg"],self.pre_ct),tf.multiply(self.layerO["ig"],self.layerO["ct_"]),name="Output_ct")
			self.pre_ct=self.layerO["ct"]
			self.layerO["tan_ct"]=tf.tanh(self.layerO["ct"],name="Output_tan_ct")
			self.layerO["ht"]=tf.multiply(self.layerO["og"],self.layerO["tan_ct"],name="Output_ht")
			self.layerOutputs["ht"]=self.layerO["ht"]
		self.loss=self.layerO["ht"]*tf.log(self.input_labels)
		#self.losses.append(self.loss)

		return(self.layerOutputs)

	def Optimize(self):
		self.layerOutputs=self.Activation()
		with tf.name_scope("Optimization_cross_entropy"):
		   self.cross_entropy=tf.reduce_mean(-tf.reduce_sum(tf.add(self.layerOutputs["ht"]*tf.log(self.Datalabels),self.Total_loss),reduction_indices=[0]),name="cross_entropy")
		   #self.cross_entropy=tf.reduce_mean(-tf.reduce_sum(self.losses,reduction_indices=[0]),name="cross_entropy")
		   #self.cross_entropy=tf.reduce_mean(-tf.reduce_sum(self.losses,reduction_indices=[0]),name="cross_entropy")
		   #OR
		   #self.cross_entropy=tf.reduce_mean(-tf.reduce_sum(self.dataresults*tf.log(self.layerO),reduction_indices=[1]))
		   #print(self.cross_entropy)
		   self.train=tf.train.GradientDescentOptimizer(0.05).minimize(self.cross_entropy,var_list=neuralNet.variable_list,name="Optimizer")
		   #print(neuralNet.variable_list)

		   neuralNet.summaries["scalar"].append({"scope":"Optimization_cross_entropy","name":"cross_entropy","value":self.cross_entropy})
		   #neuralNet.summaries["scalar"].append({"scope":"Optimization_cross_entropy","name":"Optimizer","value":self.train})

	def Train_model(self,model):
		self.Optimize()
		self.Accuracy()
		#self.layerOutputs=self.Activation()
		print("Activation","\n")
		self.summarize()
		print("Summarize","\n")
		self.graph_writer.add_graph(self.session.graph)
		self.best_accuracy=0; #Best Accuracy 
		iter=1
		for i in range(1,self.epoch+1):
			self.session.run(self.iterator.initializer)
			while True:
				try:
					self.training_set=self.session.run(self.iterator.get_next())
					print(len(self.training_set),"\n")
					feature=self.training_set["featu"]
					label=self.training_set["val"]
					print("Featu&label:",feature," ",label,"\n")

					#try:
					#self.losses=[]
					#self.Optimize()

					sessargs=[]
					for type in ["ig","fg","og","ct_"]:
						for output in self.layerOutputs[type]:
							sessargs.append(output)
					sessargs.append(self.loss)
					sessargs.append(self.merged)
					print("SessArgsCreated","\n")
					#sessargs.append(self.train)

					sessvalues=self.session.run(sessargs,feed_dict={self.Datafeatures:feature,self.Datalabels:label})
					print("sessvaluescalculated","\n")

					print("iteration: ",iter,"\t")

					if(iter%60==0):
						self.session.run(self.train,self.accuracy,feed_dict={self.Datafeatures:feature,self.Datalabels:label})
						print("Epoch: ",self.epoch,"\t","Iteration: ",iter,"\t","Accuracy: ",self.accuracy,"\n")
						print("Layer Outputs: ",sessvalues[:-3])
						
						#Check accuracy over validation set
						'''with tf.name_scope("TrainValidation"): 
							
							with tf.name_scope("ConfigValidation"):
								validation_set=self.session.run(self.viterator.get_next())
								feature=validation_set["featu"]
								label=validation_set["val"]

							with tf.name_scope("Validate"):
								accuracy=self.session.run(self.accuracy,feed_dict={self.Datafeatures:feature,self.Datalabels:label})
								print("Validation Accuracy: ",accuracy,"\n")

							with tf.name_scope("EvaluateTraining"):
								if(accuracy>=self.best_accuracy):
									self.best_accuracy=accuracy
									pass'''

						self.Total_loss=tf.zeros([1,neuralNet.classes],name="Total_loss")
						print("Optimization Done","\n")

					for type in ["ig","fg","og","ct_"]:
						self.layerOutputsVal[type]=[]
						init=1
						for index in range(init,init+neuralNet.no_hlayer+2):
							self.layerOutputsVal[type].append(sessvalues[index-1])

						self.update_feeds(type)
						print("Feed Updation Done of ",type,"\n")
						init=init+init+neuralNet.no_hlayer+2

					#print(sessvalues[-2],"\n")
					self.Total_loss=tf.add(self.Total_loss,sessvalues[-2])
					print(self.Total_loss)
					train_summary=sessvalues[-1]
					self.train_writer.add_summary(train_summary,iter)

					#print(self.losses,"\n")
					self.saver.save(self.session,"../data/tensorboard/variable_tensor"+model+".ckpt",global_step=iter)
					iter=iter+1

					'''except Exception as e:
						print (e," ",iter,"\n")'''

				except Exception as e:
					print(e," ",iter,"\n")
					break

	def Test_model(self,model):
		with tf.name_scope("TestInit"):
			self.layerOutputs=self.Activation()
			self.Accuracy()
			self.summarize()

		with tf.name_scope("Restore"):
			self.saver.restore(self.session,"../data/tensorboard/variable_tensor/"+model+".ckpt-68")

		with tf.name_scope("Testing"):
			self.session.run(self.iterator.initializer)
			iter=1
			while True:
				try:
					self.testing_set=self.session.run(self.iterator.get_next())
					print(len(self.testing_set),"\n")
					feature=self.testing_set["featu"]
					label=self.testing_set["val"]
					print("Featu&label:",feature," ",label,"\n")

					sessargs=[]
					for type in ["ig","fg","og","ct_"]:
						for output in self.layerOutputs[type]:
							sessargs.append(output)
					sessargs.append(self.accuracy)
					#sessargs.append(self.loss)
					sessargs.append(self.merged)
					print("SessArgsCreated","\n")

					sessvalues=self.session.run(sessargs,feed_dict={self.Datafeatures:feature,self.Datalabels:label})
					print("sessvaluescalculated","\n")

					print("Accuracy Testing",sessvalues[-2],"\n")

					print("iteration: ",iter,"\n")

					'''if(iter%30==0):
						self.session.run(self.train,feed_dict={self.Datafeatures:feature,self.Datalabels:label})
						self.Total_loss=tf.zeros([1,neuralNet.classes],name="Total_loss")
						print("Optimization Done","\n")'''

					for type in ["ig","fg","og","ct_"]:
						self.layerOutputsVal[type]=[]
						init=1
						for index in range(init,init+neuralNet.no_hlayer+2):
							self.layerOutputsVal[type].append(sessvalues[index-1])

						self.update_feeds(type)
						print("Feed Updation Done of ",type,"\n")
						init=init+init+neuralNet.no_hlayer+2

					#print(sessvalues[-2],"\n")
					'''self.Total_loss=tf.add(self.Total_loss,sessvalues[-2])
					print(self.Total_loss)'''

					test_summary=sessvalues[-1]
					self.test_writer.add_summary(test_summary,iter)

					iter=iter+1

				except Exception as e:
					print(e," ",iter,"\n")
					break

	def Accuracy(self):
		with tf.name_scope("Accurate_predictiction"):
			correct_prediction=tf.equal(tf.argmax(self.layerOutputs["ht"],1), tf.argmax(self.Datalabels,1),name="Prediction")
			self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32),name="Accuracy")

			#neuralNet.summaries["scalar"].append({"scope":"Accurate_predictiction","name":"Prediction","value":correct_prediction})
			neuralNet.summaries["scalar"].append({"scope":"Accurate_predictiction","name":"Accuracy","value":self.accuracy})

	def summarize(self):
		self.saver=tf.train.Saver(neuralNet.variable_list)
		for typ,summaries in neuralNet.summaries.items():
			for summary in summaries:
				with tf.name_scope(summary["scope"]):
					method="tf.summary."+typ+"(a,b)"
					eval(method,{"tf":tf},{"a":summary["name"],"b":summary["value"]})
		self.merged=tf.summary.merge_all()


'''neuralNet.Layersconf()
n1=neuralNet(tf.Session())
n1.input_features=n1.Datafeatures
n1.input_labels=n1.Datalabels
n1.Train_model()'''