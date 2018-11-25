from pyspark import SparkContext
import sys
from subprocess import Popen
import os
import datetime

#Kernel
from kernel import K
#TIMESTAMP
st = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#SET INFO
print(st+" Setting Parameter")
data_input = str(sys.argv[1])
max_iteration = int(sys.argv[2])
cluster_num = int(sys.argv[3])
data_output = str(sys.argv[4])
#Create Spark Context
print(st+" Creating Spark Context")
sc = SparkContext(str(sys.argv[5]))
#create temp folder
#sc.setCheckpointDir("TEMP")
#Read data input
print(st+" Read Data Input")
data = sc.textFile(data_input)

#Add random class to each data
#Also add index
print(st+" Adding index and random class to each data")
cluster = 0;
index = -1;
def class_random(x):
	global cluster
	global cluster_num
	global index
	cluster = cluster +1 if cluster<cluster_num else 1
	index = index +1
	return (index, [float(num) for num in x.strip().split(',')], cluster) 

data = data.map(class_random)

#ITER
for i in range(max_iteration):
	#Counting cluster member
	print(st+" Iteration "+str(i+1))
	Cluster_num = data.map(lambda x: (x[2],1)).reduceByKey(lambda x,y:x+y)
	#Kernel Matrix

	KM = data.cartesian(data).map(lambda x : ((x[0][0], x[1][0]),K(x[0][1],x[1][1]), (x[0][2], x[1][2])))
	#Calculating B

	B = KM.map(lambda x: ((x[0][0],x[2][1]), x[1])).reduceByKey(lambda x,y:x+y).map(lambda x: (x[0][1], (x[0][0], x[1]))).join(Cluster_num).map(lambda x: (x[0], (x[1][0][0], x[1][0][1]/x[1][1]*-2)))
	#Calculating C

	C = KM.filter(lambda x : x[2][0] == x[2][1]).map(lambda x : (x[2][0], x[1])).reduceByKey(lambda x,y:x+y).join(Cluster_num).map(lambda x : (x[0], x[1][0]/pow(x[1][1],2)))
	#Alocating cluster member

	data = B.join(C).map(lambda x: (x[1][0][0], (x[1][0][1]+x[1][1], x[0]))).reduceByKey(lambda x, y : (min(x[0],y[0]),x[1] if x[0]<y[0] else y[1])).map(lambda x:(x[0],x[1][1])).join(data).map(lambda x:(x[0], x[1][1], x[1][0]))
	data = sc.parallelize(data.collect())
	print(data.toDebugString())

#Writing result
print(st+" Writing result")
result = data.map(lambda x:(x[1],x[2])).sortBy(lambda x : x[0])
result.saveAsTextFile(data_output)


#
'''
print(data.collect())
print(B.collect())
print(C.collect())
print(B.join(C).map(lambda x: (x[1][0][0], (x[1][0][1]+x[1][1], x[0]))).collect())
print(result.collect())
'''
#.reduceByKey(lambda x, y:x+y).sortBy(lambda x:x[0])



'''
#Kernel Matrix
print(st+" Calculating Kernel Matrix")
KM = data.cartesian(data).map(lambda x: ((x[0][0], x[1][0]),(K(x[0][1]['feature'],x[1][1]['feature'])*x[0][1]['class']*x[1][1]['class'])))

#Write support vector to textfile
print(st+" Writing Support Vector")
result.saveAsTextFile(data_output+"/support_vector")

'''
