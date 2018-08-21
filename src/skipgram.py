import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import numpy.matlib

class neuralNet:
	vocabulary_size=0
	embedding_size=0
	window_size=0
	layers=[]
	layer_schema={"neurons":"", "weights":" ", "biases":" "}
	inputwidth=4
	classes=2
	no_hlayer=2
	neurons_list=[inputwidth,[1550,2000]]
	variable_list={}
	summaries={"histogram":[],"scalar":[]}
	training_set=[[[1.,2.,3.,4.],[8.,6.]],[[2.3,7.6,3.9,0.3],[9.9,5.9]],[[2.3,7.6,3.9,0.3],[9.9,5.9]],[[2.3,7.6,3.9,0.3],[9.9,5.9]],[[2.3,7.6,3.9,0.3],[9.9,5.9]],[[2.3,7.6,3.9,0.3],[9.9,5.9]],[[2.3,7.6,3.9,0.3],[9.9,5.9]],[[2.3,7.6,3.9,0.3],[9.9,5.9]],[[2.3,7.6,3.9,0.3],[9.9,5.9]],[[2.3,7.6,3.9,0.3],[9.9,5.9]],[[2.3,7.6,3.9,0.3],[9.9,5.9]],[[2.3,7.6,3.9,0.3],[9.9,5.9]],[[2.3,7.6,3.9,0.3],[9.9,5.9]]]
	testing_set=[]
	epoch=5

	def __init__(self):
		with tf.name_scope("Datainput"):
			self.features=tf.placeholder(tf.float32,[None,self.inputwidth],name="feature_set")
			self.labels=tf.placeholder(tf.float32,[None,self.classes],name="labels")

		with tf.name_scope("Embeddings"):
			self.embeddings=tf.Variable(tf.random_uniform([neuralNet.vocabulary_size,neuralNet.embedding_size],-1.0,1.0),name="embeddings")
			self.variable_list["embeddings"]=self.embeddings

		self.session=tf.Session()
		self.init=tf.global_variables_initializer()
		self.graph_writer=tf.summary.FileWriter("../data/tensorboard")
		self.train_writer=tf.summary.FileWriter("../data/tensorboard")
		self.test_writer=tf.summary.FileWriter("../data/tensorboard")
		self.session.run(self.init)

	@classmethod
	def Layersconf(cls):
		cls.Inputlayer(cls.neurons_list[0])
		cls.Hiddenlayers(cls.neurons_list[1])
		cls.Outputlayer()
		print("conf Done")
		print(cls.layers)

	@classmethod
	def Inputlayer(cls,neurons):
		cls.layers.append({"neurons":neurons})
		with tf.name_scope("Inputlayer"):
			cls.layers[0]["weights"] = tf.Variable(tf.random_normal([cls.inputwidth,cls.layers[0]["neurons"]]),tf.float32,name="weights")
			cls.variable_list["weights"+str(0)]=cls.layers[0]["weights"]

			cls.layers[0]["biases"] = tf.Variable(tf.zeros([1,cls.layers[0]["neurons"]]),tf.float32,name="biases")
			cls.variable_list["biases"+str(0)]=cls.layers[0]["biases"]

			cls.summaries["histogram"].append({"scope":"Inputlayer","name":"weights","value":cls.layers[0]["weights"]})
			cls.summaries["histogram"].append({"scope":"Inputlayer","name":"biases","value":cls.layers[0]["biases"]})
		cls.layer_no_out=cls.layers[0]["neurons"]

	@classmethod
	def Hiddenlayers(cls,neurons_list):
		cls.layers.append([])
		for index , neurons in enumerate(neurons_list):
			cls.layers[1].append({"neurons":neurons})
			with tf.name_scope("Hiddenlayer"+str(index)):
				cls.layers[1][index]["weights"] = tf.Variable(tf.random_normal([cls.layer_no_out,cls.layers[1][index]["neurons"]]),tf.float32,name="weights")
				cls.variable_list["weights"+str(1)+str(index)]=cls.layers[1][index]["weights"]

				cls.layers[1][index]["biases"] = tf.Variable(tf.ones([1,cls.layers[1][index]["neurons"]]),tf.float32,name="biases")
				cls.variable_list["biases"+str(1)+str(index)]=cls.layers[1][index]["biases"]

				cls.summaries["histogram"].append({"scope":"Hiddenlayer"+str(index),"name":"weights","value":cls.layers[1][index]["weights"]})
				cls.summaries["histogram"].append({"scope":"Hiddenlayer"+str(index),"name":"biases","value":cls.layers[1][index]["biases"]})

			cls.layer_no_out=cls.layers[1][index]["neurons"]

	@classmethod
	def Outputlayer(cls):
		cls.layers.append({"neurons":cls.classes})
		with tf.name_scope("Outputlayer"):
			cls.layers[cls.no_hlayer]["weights"] = tf.Variable(tf.random_normal([cls.layer_no_out,cls.classes]),tf.float32,name="weights")
			cls.variable_list["weights"+str(2)]=cls.layers[2]["weights"]

			cls.layers[cls.no_hlayer]["biases"] = tf.Variable(tf.ones([1,cls.classes]),tf.float32,name="biases")
			cls.variable_list["biases"+str(2)]=cls.layers[2]["biases"]

			cls.summaries["histogram"].append({"scope":"Outputlayer","name":"weights","value":cls.layers[2]["weights"]})
			cls.summaries["histogram"].append({"scope":"Outputlayer","name":"biases","value":cls.layers[2]["biases"]})

	def Activation(self):
		with tf.name_scope("Inputlayer"):
			self.layerI=tf.add(tf.matmul(self.features,neuralNet.layers[0]["weights"]),neuralNet.layers[0]["biases"],name="Weightedinput")
			#self.layerO=tf.nn.sigmoid(self.layerI,name="Outputsigmoid")
			self.layerO=tf.nn.relu(self.layerI,name="Outputrelu")

			neuralNet.summaries["histogram"].append({"scope":"Inputlayer","name":"Weightedinput","value":self.layerI})
			neuralNet.summaries["histogram"].append({"scope":"Inputlayer","name":"Outputrelu","value":self.layerO})

		for index, layer in enumerate(neuralNet.layers[1]):
			with tf.name_scope("Hiddenlayer"+str(index)):
				self.layerI=tf.add(tf.matmul(self.layerO,layer["weights"]),layer["biases"],name="Weightedinput")
				#self.layerO=tf.nn.sigmoid(self.layerI,name="Outputsigmoid")
				self.layerO=tf.nn.relu(self.layerI,name="Outputrelu")

				neuralNet.summaries["histogram"].append({"scope":"Hiddenlayer"+str(index),"name":"Weightedinput","value":self.layerI})
				neuralNet.summaries["histogram"].append({"scope":"Hiddenlayer"+str(index),"name":"Outputrelu","value":self.layerO})

		with tf.name_scope("Outputlayer"):
			self.layerI=tf.add(tf.matmul(self.layerO,neuralNet.layers[2]["weights"]),neuralNet.layers[2]["biases"],name="Weightedinput")
			self.layerO=tf.nn.softmax(self.layerI,name="Outputsoftmax")

			neuralNet.summaries["histogram"].append({"scope":"Outputlayer","name":"Weightedinput","value":self.layerI})
			neuralNet.summaries["histogram"].append({"scope":"Outputlayer","name":"Outputrelu","value":self.layerO})

		self.embed= tf.nn.embedding_lookup(self.embeddings,self.features)

		#return(self.layerO)
		return(self.embed)

	def Optimize(self):
		with tf.name_scope("Optimization_cross_entropy"):
			self.embed=self.Activation()
			self.cross_entropy=tf.nn.nce_loss(weights)
			#self.cross_entropy=tf.reduce_mean(-tf.reduce_sum(self.labels*tf.log(self.layerO),reduction_indices=[1]),name="cross_entropy")
			#OR
			#self.cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.layerO,self.dataresults),name="cross_entropy")
			print(self.cross_entropy)
			self.train=tf.train.GradientDescentOptimizer(0.005).minimize(self.cross_entropy,name="Optimizer")
			neuralNet.summaries["scalar"].append({"scope":"Optimization_cross_entropy","name":"cross_entropy","value":self.cross_entropy})
			#neuralNet.summaries["scalar"].append({"scope":"Optimization_cross_entropy","name":"Optimizer","value":self.train})

	def Accuracy(self):
		with tf.name_scope("Accurate_predictiction"):
			correct_prediction=tf.equal(tf.argmax(self.layerO,1), tf.argmax(self.labels,1),name="Prediction")
			self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32),name="Accuracy")
			#neuralNet.summaries["scalar"].append({"scope":"Accurate_predictiction","name":"Prediction","value":correct_prediction})
			neuralNet.summaries["scalar"].append({"scope":"Accurate_predictiction","name":"Accuracy","value":self.accuracy})

	def Train_model(self):
		self.Optimize()
		self.summarize()
		self.graph_writer.add_graph(self.session.graph)
		factor=int(len(self.training_set)/self.epoch)
		print(len(self.training_set))
		print(factor)
		if(factor!=0):
			for i in range(1,self.epoch+1):
				batch=self.training_set[(i-1)*factor:i*factor]
				#print(batch)
				feature_set=[]
				label_set=[]
				for feature,label in batch:
					feature_set.append(feature)
					label_set.append(label)
				train_summary,_=self.session.run([self.merged,self.train],feed_dict={self.features:feature_set,self.labels:label_set})
				self.train_writer.add_summary(train_summary,i)
				self.saver.save(self.session,"../data/pickle/variable_tensor/",global_step=i)

	def Test_model(self):
			feature_set=[]
			label_set=[]
			for feature,label in self.testing_set:
				feature_set.append(feature)
				label_set.append(label)
			accuracy,test_summary=self.session.run([self.accuracy,self.merged],feed_dict={self.features:feature_set,self.labels:label_set})
			self.test_writer.add_summary(test_summary,1)
			print("Accuracy for test batch: ",accuracy)

	def summarize(self):
		for typ,summaries in neuralNet.summaries.items():
			for summary in summaries:
				with tf.name_scope(summary["scope"]):
					method="tf.summary."+typ+"(a,b)"
					eval(method,{"tf":tf},{"a":summary["name"],"b":summary["value"]})
		self.merged=tf.summary.merge_all()
		print(neuralNet.variable_list["embeddings"])
		self.saver=tf.train.Saver(neuralNet.variable_list["embeddings"])
