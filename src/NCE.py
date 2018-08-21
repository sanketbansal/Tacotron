import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import numpy.matlib

class neuralNet:
	vocabulary_size=0
	embedding_size=0
	sample_size=0
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
			self.embeddings=tf.Variable(tf.random_uniform([neuralNet.vocabulary_size,neuralNet.embedding_size]-1.0,1.0),name="embeddings")
			self.variable_list["embeddings"]=self.embeddings

        with tf.name("NCE_var"):
            self.nce_weights=tf.Variable([neuralNet.vocabulary_size,neuralNet.embedding_size],name="nce_weights")
            self.nce_biases=tf.Variable(tf.zeros([neuralNet.vocabulary_size]),name="nce_biases")

		self.session=tf.Session()
		self.init=tf.global_variables_initializer()
		self.graph_writer=tf.summary.FileWriter("../data/tensorboard")
		self.train_writer=tf.summary.FileWriter("../data/tensorboard")
		self.test_writer=tf.summary.FileWriter("../data/tensorboard")
		self.session.run(self.init)

	def Activation(self):
		embed= tf.nn.embedding_lookup(self.embeddings,self.features)
		return(embed)

	def Optimize(self):
		with tf.name_scope("Optimization_cross_entropy"):
			self.embed=self.Activation()
			self.cross_entropy=tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weights,biases=self.nce_biases,labels=self.labels,inputs=self.embed,num_sampled=neuralNet.sample_size,num_classes=neuralNet.vocabulary_size),name="cross_entropy")

			print(self.cross_entropy)
			self.train=tf.train.GradientDescentOptimizer(0.005).minimize(self.cross_entropy,name="Optimizer")

			neuralNet.summaries["scalar"].append({"scope":"Optimization_cross_entropy","name":"cross_entropy","value":self.cross_entropy})
			#neuralNet.summaries["scalar"].append({"scope":"Optimization_cross_entropy","name":"Optimizer","value":self.train})

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

	def summarize(self):
		for typ,summaries in neuralNet.summaries.items():
			for summary in summaries:
				with tf.name_scope(summary["scope"]):
					method="tf.summary."+typ+"(a,b)"
					eval(method,{"tf":tf},{"a":summary["name"],"b":summary["value"]})
		self.merged=tf.summary.merge_all()
		print(neuralNet.variable_list)
		self.saver=tf.train.Saver(neuralNet.variable_list)
