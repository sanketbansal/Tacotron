import tensorflow as tf
import numpy as np
import numpy.matlib
import copy

class neuralNet:

	layers={}
	layer_schema={ "weights":" ", "biases":" "}
	conv_arch=[2,2]
	inputshape=[32,32,3]
	filtersize=[3,3]	
	no_filters=64
	poolingsize=[1,2,2,1]
	stride=1
	variable_list={}
	summaries={"histogram":[],"scalar":[]}
	outputs={}
	input_features=0
	input_labels=0


	def __init__(self,session):
		self.session=session
		self.Layersconf()


	def Layersconf(self):
		self.conv_input=copy.deepcopy(self.filtersize)
		self.conv_input.append(self.inputshape[2])
		for value in range(0,self.conv_arch[1]):
			self.Convlayer(self.conv_input,0,value)
			#self.Relulayers(0,value)
			self.conv_input=copy.deepcopy(self.filtersize)
			self.conv_input.append(self.conv_out)
			for index in range(1,self.conv_arch[0]):
				self.Convlayer(self.conv_input,index,value)
				#self.Relulayers(index,value)
				self.conv_input=copy.deepcopy(self.filtersize)
				self.conv_input.append(self.conv_out)
			self.MAXPoollayer(value)

		self.convflatO=tf.contrib.layers.flatten(self.outputs["pool"+str(self.conv_arch[1]-1)],scope="ConvFlatO")
		print(self.convflatO,"\n")

		self.summarize()

		print("conf Done")
		print(self.layers,"\n")


	def Convlayer(self,input,index,index1):
		input.append(self.session.run(tf.random_uniform([],30,80,tf.int32)))
		print(input,"\n")
		with tf.name_scope("Convlayer"+str(index)+str(index1)):
			self.layers["conv"+str(index)+str(index1)]={}
			self.layers["conv"+str(index)+str(index1)]["weights"] =tf.Variable(tf.random_normal(input),tf.float32,name="weights")
			#self.layers["conv"+str(index)+str(index1)]["weights"] =tf.get_variable("weights",input)
			self.variable_list["conv"+str(index)+str(index1)+"weights"]=self.layers["conv"+str(index)+str(index1)]["weights"]

			self.layers["conv"+str(index)+str(index1)]["biases"] = tf.Variable(tf.ones([1,self.no_filters]),tf.float32,name="biases")
			self.variable_list["conv"+str(index)+str(index1)+"biases"]=self.layers["conv"+str(index)+str(index1)]["biases"]

			self.summaries["histogram"].append({"scope":"Convlayer"+str(index)+str(index1),"name":"weights","value":self.layers["conv"+str(index)+str(index1)]["weights"]})
			self.summaries["histogram"].append({"scope":"Convlayer"+str(index)+str(index1),"name":"biases","value":self.layers["conv"+str(index)+str(index1)]["biases"]})


			with tf.name_scope("Activation"):
				ConvI=tf.nn.conv2d(self.input_features,self.layers["conv"+str(index)+str(index1)]["weights"],strides=[1,1,1,1],padding='SAME')
				#self.outputs["conv"+str(index)+str(index1)]=tf.nn.relu(ConvI,self.layers["conv"+str(index)+str(index1)]["biases"],name="ConvReluOUT"+str(index)+str(index1))
				self.outputs["conv"+str(index)+str(index1)]=tf.nn.relu(ConvI,name="ConvReluOUT")
				self.input_features=self.outputs["conv"+str(index)+str(index1)]
			self.conv_out=input[-1]


	'''def Relulayer(self,index,index1):
		with tf.name_scope("Relulayer"+str(index)+str(index1)):
			self.layers["relu"+str(index)+str(index1)]={}
			self.layers["relu"+str(index)+str(index1)]["weights"] = tf.Variable(tf.random_normal([self.conv_out_shape,self.layers[1][index]["neurons"]]),tf.float32,name="weights")
			self.variable_list["relu"+str(index)+str(index1)+"weights"]=self.layers["relu"+str(index)+str(index1)]["weights"]

			self.layers["relu"+str(index)+str(index1)]["biases"] = tf.Variable(tf.ones([1,self.layers[1][index]["neurons"]]),tf.float32,name="biases")
			self.variable_list["relu"+str(index)+str(index1)+"biases"]=self.layers["relu"+str(index)+str(index1)]["biases"]

			self.summaries["histogram"].append({"scope":"Relulayer"+str(index)+str(index1),"name":"weights","value":self.layers["relu"+str(index)+str(index1)]["weights"]})
			self.summaries["histogram"].append({"scope":"Relulayer"+str(index)+str(index1),"name":"biases","value":self.layers["relu"+str(index)+str(index1)]["biases"]})
			
			self.relu_out_shape=self.layers[1][index]["neurons"]'''

	def MAXPoollayer(self,index):
		with tf.name_scope("MAXPoollayer"+str(index)):
			self.outputs["pool"+str(index)]=tf.nn.max_pool(self.input_features,self.poolingsize,[1,1,1,1],"SAME",name="Maxpool")
			#self.outputs["conv"+str(self.conv_arch[0]-1)+str(index)]
			#self.pool_out_shape=self.outputs["pool"+str(index)].shape()
			self.input_features=self.outputs["pool"+str(index)]


	def Activation(self):
		with tf.name_scope("ConvActivation"):
			with tf.name_scope("Convlayer"):
				self.layerI=tf.add(tf.matmul(self.input_features,neuralNet.layers[0]["weights"]),neuralNet.layers[0]["biases"],name="Weightedinput")
				#self.layerO=tf.nn.sigmoid(self.layerI,name="Outputsigmoid")
				self.layerO=tf.nn.relu(self.layerI,name="Outputrelu")

				neuralNet.summaries["histogram"].append({"scope":"Inputlayer","name":"Weightedinput","value":self.layerI})
				neuralNet.summaries["histogram"].append({"scope":"Inputlayer","name":"Outputrelu","value":self.layerO})
				#print(self.layerI)

			for index, layer in enumerate(neuralNet.layers[1]):
				with tf.name_scope("Hiddenlayer"+str(index)):
					self.layerI=tf.add(tf.matmul(self.layerO,layer["weights"]),layer["biases"],name="Weightedinput")
					#self.layerO=tf.nn.sigmoid(self.layerI,name="Outputsigmoid")
					self.layerO=tf.nn.relu(self.layerI,name="Outputrelu")

					neuralNet.summaries["histogram"].append({"scope":"Hiddenlayer"+str(index),"name":"Weightedinput","value":self.layerI})
					neuralNet.summaries["histogram"].append({"scope":"Hiddenlayer"+str(index),"name":"Outputrelu","value":self.layerO})
					#print(self.layerI)

			with tf.name_scope("Outputlayer"):
				self.layerI=tf.add(tf.matmul(self.layerO,neuralNet.layers[2]["weights"]),neuralNet.layers[2]["biases"],name="Weightedinput")
				self.layerO=tf.nn.softmax(self.layerI,name="Outputsoftmax")

				neuralNet.summaries["histogram"].append({"scope":"Outputlayer","name":"Weightedinput","value":self.layerI})
				neuralNet.summaries["histogram"].append({"scope":"Outputlayer","name":"Outputrelu","value":self.layerO})
				#print(self.layerI,"\n")

			return(self.layerO)
	
	def summarize(self):
		for typ,summaries in neuralNet.summaries.items():
			for summary in summaries:
				with tf.name_scope(summary["scope"]):
					method="tf.summary."+typ+"(a,b)"
					eval(method,{"tf":tf},{"a":summary["name"],"b":summary["value"]})

#print(tf.Session().run(tf.random_normal([2,32,32,3],name="datafeatures")))
'''neuralNet.input_features=tf.random_normal([1000,32,32,3],name="datafeatures")
neuralNet.input_labels=tf.random_normal([1,10],name="datalabels")
neuralNet.Layersconf(tf.Session())
#conv1=neuralNet(tf.Session())'''
