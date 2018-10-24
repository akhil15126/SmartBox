import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def analyse_helper(sess,model,X_test,Y_test,classes,CMC):
	rank=[]
	
	x=tf.placeholder('float', shape=X_test.shape)
	prediction=model.get_logits(x)
	scores=tf.nn.softmax(prediction)
	scores=sess.run(scores, feed_dict={x:X_test})
	for s,l in zip(scores,Y_test):
		tmp=np.argsort(s)
		l=np.argmax(l)
		b=list(reversed(list(tmp)))
		rank.append(b.index(l)+1)
	X_rank=list(np.arange(classes)+1)
	Y_rank=[0]*len(X_rank)
	for i in range(0,len(Y_rank)):
		Y_rank[i]=rank.count(i+1)

	for i in range(1,len(Y_rank)):
		Y_rank[i]+=Y_rank[i-1]
	Y_rank=list((np.array(Y_rank))/len(Y_test))
	# print(X_rank)
	# print(Y_rank)
	plt.figure(0)
	a,=plt.plot(X_rank,Y_rank)
	CMC.append(a)

	pred=model.model.predict(X_test)
	pred=np.argmax(pred,axis=1)
	actual=np.argmax(Y_test,axis=1)
	return np.mean(np.equal(pred,actual))


def identification_report(sess,model,classes,X_test,Y_test,adv=None,mit=None,save='./cmc.jpg'):
	report={}

	CMC=[]
	leg=[]
	report["Identification Accuracy (before attack)"]=analyse_helper(sess,model,X_test,Y_test,classes,CMC)
	leg.append("Before Attack")
	if(adv is not None):
		report["Identification Accuracy (after attack)"]=analyse_helper(sess,model,adv,Y_test,classes,CMC)
		leg.append("After Attack")

	if(mit is not None):
		if type(mit) is np.ndarray:
			report["Identification Accuracy (after mitigation)"]=analyse_helper(sess,model,mit,Y_test,classes,CMC)	
		else:
			report["Identification Accuracy (after mitigation)"]=analyse_helper(sess,mit,adv,Y_test,classes,CMC)
		leg.append("After Mitigation")
		
	plt.figure(0)
	plt.xlabel("Rank")
	plt.ylabel("Accuracy")
	plt.title("CMC curve")
	plt.legend(tuple(CMC), tuple(leg), loc='lower right')
	plt.savefig(save)	
	plt.close()
	return report


def print_identification_report(sess,model,classes,X_test,Y_test,adv=None,mit=None,save='./cmc.jpg'):
	report=identification_report(sess,model,classes,X_test,Y_test,adv,mit,save)
	print("\n")
	print("------------------------ IDENTIFICATION REPORT ------------------------")
	print("\n")

	for k,v in report.items():
		print(k+" : "+str(v*100)+"%")
	print("CMC Location : "+save)

	print("\n")
	print("-----------------------------------------------------------------------")
	print("\n")

	return report
