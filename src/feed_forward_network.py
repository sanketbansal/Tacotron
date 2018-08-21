import tensorflow as tf
import numpy as np
import numpy.matlib

class neuralNet:

	layers=[]
	layer_schema={"neurons":"", "weights":" ", "biases":" "}
	inputwidth=784
	classes=10
	no_hlayer=1
	neurons_list=[inputwidth,[1000]]
	variable_list={}
	summaries={"histogram":[],"scalar":[]}
	training_set={}
	testing_set={}
	input_features=0
	input_labels=0
	iterator=None
	epoch=1

	def __init__(self,session):
		with tf.name_scope("Datainput"):
			self.Datafeatures=tf.placeholder(tf.float32,[None,self.inputwidth],name="feature_set")
			self.Datalabels=tf.placeholder(tf.float32,[None,self.classes],name="labels")
		self.session=session
		self.init=tf.global_variables_initializer()
		self.graph_writer=tf.summary.FileWriter("../data/tensorboard/summary/graph/")
		self.train_writer=tf.summary.FileWriter("../data/tensorboard/summary/train/")
		self.test_writer=tf.summary.FileWriter("../data/tensorboard/summary/test/")
		self.session.run(self.init)

	@classmethod
	def Layersconf(cls):
		cls.Inputlayer(cls.neurons_list[0])
		cls.Hiddenlayers(cls.neurons_list[1])
		cls.Outputlayer()
		print("conf Done")
		print(cls.layers,"\n")

	@classmethod
	def Inputlayer(cls,neurons):
		cls.layers.append({"neurons":neurons})
		with tf.name_scope("Inputlayer"):
			cls.layers[0]["weights"] = tf.Variable(tf.random_normal([cls.inputwidth,cls.layers[0]["neurons"]]),dtype=tf.float32,name="weights")
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
				cls.layers[1][index]["weights"] = tf.Variable(tf.random_normal([cls.layer_no_out,cls.layers[1][index]["neurons"]]),dtype=tf.float32,name="weights")
				cls.variable_list["weights"+str(1)+str(index)]=cls.layers[1][index]["weights"]

				cls.layers[1][index]["biases"] = tf.Variable(tf.zeros([1,cls.layers[1][index]["neurons"]]),tf.float32,name="biases")
				cls.variable_list["biases"+str(1)+str(index)]=cls.layers[1][index]["biases"]

				cls.summaries["histogram"].append({"scope":"Hiddenlayer"+str(index),"name":"weights","value":cls.layers[1][index]["weights"]})
				cls.summaries["histogram"].append({"scope":"Hiddenlayer"+str(index),"name":"biases","value":cls.layers[1][index]["biases"]})

			cls.layer_no_out=cls.layers[1][index]["neurons"]

	@classmethod
	def Outputlayer(cls):
		cls.layers.append({"neurons":cls.classes})
		with tf.name_scope("Outputlayer"):
			cls.layers[2]["weights"] = tf.Variable(tf.random_normal([cls.layer_no_out,cls.classes]),dtype=tf.float32,name="weights")
			cls.variable_list["weights"+str(2)]=cls.layers[2]["weights"]

			cls.layers[2]["biases"] = tf.Variable(tf.zeros([1,cls.classes]),tf.float32,name="biases")
			cls.variable_list["biases"+str(2)]=cls.layers[2]["biases"]

			cls.summaries["histogram"].append({"scope":"Outputlayer","name":"weights","value":cls.layers[2]["weights"]})
			cls.summaries["histogram"].append({"scope":"Outputlayer","name":"biases","value":cls.layers[2]["biases"]})

	def Activation(self):
		with tf.name_scope("Inputlayer"):
			self.layerI=tf.add(tf.matmul(self.input_features,neuralNet.layers[0]["weights"]),neuralNet.layers[0]["biases"],name="Weightedinput")
			self.layerI=tf.layers.batch_normalization(self.layerI)
			
			#self.layerO=tf.nn.sigmoid(self.layerI,name="Outputsigmoid")
			self.layerO=tf.nn.relu(self.layerI,name="Outputrelu")
			#self.layerO=tf.nn.elu(self.layerI,name="Outputelu")
			#self.layerO=tf.nn.tanh(self.layerI,name="Outputtanh")
			#self.layerO=self.layerI

			neuralNet.summaries["histogram"].append({"scope":"Inputlayer","name":"Weightedinput","value":self.layerI})
			neuralNet.summaries["histogram"].append({"scope":"Inputlayer","name":"Outputrelu","value":self.layerO})
			#print(self.layerI)

		for index, layer in enumerate(neuralNet.layers[1]):
			with tf.name_scope("Hiddenlayer"+str(index)):
				self.layerI=tf.add(tf.matmul(self.layerO,layer["weights"]),layer["biases"],name="Weightedinput")
				self.layerI=tf.layers.batch_normalization(self.layerI)
				
				# self.layerO=tf.nn.sigmoid(self.layerI,name="Outputsigmoid")
				self.layerO=tf.nn.relu(self.layerI,name="Outputrelu")
				#self.layerO=tf.nn.elu(self.layerI,name="Outputelu")
				#self.layerO=tf.nn.tanh(self.layerI,name="Outputtanh")
				#self.layerO=self.layerI

				neuralNet.summaries["histogram"].append({"scope":"Hiddenlayer"+str(index),"name":"Weightedinput","value":self.layerI})
				neuralNet.summaries["histogram"].append({"scope":"Hiddenlayer"+str(index),"name":"Outputrelu","value":self.layerO})
				#print(self.layerI)

		with tf.name_scope("Outputlayer"):
			self.layerI=tf.add(tf.matmul(self.layerO,neuralNet.layers[2]["weights"]),neuralNet.layers[2]["biases"],name="Weightedinput")
			self.layerI=tf.layers.batch_normalization(self.layerI)
			
			#self.layerO=tf.nn.sigmoid(self.layerI,name="Outputsigmoid")
			self.layerO=tf.nn.softmax(self.layerI,name="Outputsoftmax")
			#self.layerO=tf.nn.tanh(self.layerI,name="Outputtanh")
			#self.layerO=tf.nn.relu(self.layerI,name="Outputrelu")
			#self.layerO=self.layerI

			neuralNet.summaries["histogram"].append({"scope":"Outputlayer","name":"Weightedinput","value":self.layerI})
			neuralNet.summaries["histogram"].append({"scope":"Outputlayer","name":"Outputsoftmax","value":self.layerO})
			#print(self.layerI,"\n")
		return(self.layerO)

	def Optimize(self):
		self.layerO=self.Activation()
		with tf.name_scope("Optimization_cross_entropy"):
			#self.cross_entropy=tf.reduce_mean(-tf.reduce_sum(self.Datalabels*tf.log(self.layerO),reduction_indices=[1]),name="cross_entropy")

			#self.cross_entropy=tf.reduce_mean(tf.reduce_sum(tf.square(self.Datalabels-self.layerI),reduction_indices=[1]),name="cross_entropy")
			#OR
			self.cross_entropy=tf.reduce_mean(tf.losses.softmax_cross_entropy(self.Datalabels,self.layerI),name="cross_entropy")

			print(self.cross_entropy)
			self.train=tf.train.GradientDescentOptimizer(0.4).minimize(self.cross_entropy,name="Optimizer")
			#self.train=tf.train.AdamOptimizer(0.001).minimize(self.cross_entropy,name="Optimizer")

			neuralNet.summaries["scalar"].append({"scope":"Optimization_cross_entropy","name":"cross_entropy","value":self.cross_entropy})

	def Accuracy(self):
		with tf.name_scope("Accurate_predictiction"):
			correct_prediction=tf.equal(tf.argmax(self.layerO,1), tf.argmax(self.Datalabels,1),name="Prediction")
			self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32),name="Accuracy")
			
			#self.losses=tf.keras.losses.binary_crossentropy(self.Datalabels,self.layerO)
			#self.accuracy=tf.keras.metrics.binary_accuracy(self.Datalabels,self.layerO)

			#neuralNet.summaries["scalar"].append({"scope":"Accurate_predictiction","name":"Prediction","value":correct_prediction})
			neuralNet.summaries["scalar"].append({"scope":"Accurate_predictiction","name":"Accuracy","value":self.accuracy})

	def Train_model(self,model):
		self.Optimize()
		self.Accuracy()
		self.summarize()
		self.graph_writer.add_graph(self.session.graph)

		iter=1
		for i in range(1,self.epoch+1):
			self.session.run(self.iterator.initializer)
			while True:
				try:
					self.training_set=self.session.run(self.iterator.get_next())
					#print(len(self.training_set),"\n")
					feature_set=self.training_set["featu"]
					label_set=self.training_set["val"]

					#print(feature_set[-1])

					sessvalues=[self.merged,self.train,self.accuracy]
					#sessvalues=[]
					for op in neuralNet.summaries["histogram"]:
							if(op["name"]!="weights" and op["name"]!="biases"):
								sessvalues.append(op["value"])
					
					print(sessvalues,"\n","sessvalues created")

					self.session.run(tf.initialize_all_variables())

					output = self.session.run(sessvalues,feed_dict={self.Datafeatures:feature_set,self.Datalabels:label_set})
					print("Optimization Done ","\n")

					print(output[3:],"\n")
					print("Iteration: ",iter,"\t","output: ","","\t","Labels: ",label_set,"\t","accuracy: ",output[2],"\n")
					self.train_writer.add_summary(output[0],iter)
					self.saver.save(self.session,"../data/tensorboard/variable_tensor/"+model+".ckpt",global_step=iter)
					iter=iter+1

				except Exception as e:
					print(iter)
					#print(self.session.run(tf.get_default_graph().as_graph_def().node))
					print(e)
					break

	def Test_model(self,model):
		with tf.name_scope("RestoreVariable"):
			self.saver.restore(self.session,"../data/tensorboard/variable_tensor/"+model+".ckpt-65")

		with tf.name_scope("RunTest"):
			self.session.run(self.iterator.initializer)
			iter=0
			while True:
				try:
					self.training_set=self.session.run(self.iterator.get_next())
					print(len(self.training_set))
					feature_set=self.training_set["featu"]
					label_set=self.training_set["val"]

					test_summary,prediction=self.session.run([self.merged,self.Activation()],feed_dict={self.Datafeatures:feature_set,self.Datalabels:label_set})
					self.test_writer.add_summary(test_summary,iter)
					print(prediction)
					iter=iter+1

				except Exception as e:
					print(iter)
					break

		with tf.name_scope("MetricEvaluation"):
			#tf.metrics().accuracy(prediction)
			pass

	def summarize(self):
		for typ,summaries in neuralNet.summaries.items():
			for summary in summaries:
				with tf.name_scope(summary["scope"]):
					method="tf.summary."+typ+"(a,b)"
					eval(method,{"tf":tf},{"a":summary["name"],"b":summary["value"]})
		self.merged=tf.summary.merge_all()
		print(self.variable_list)
		self.saver=tf.train.Saver(neuralNet.variable_list)

'''neuralNet.Layersconf()
neural1=neuralNet()
neural1.Train_model()'''
