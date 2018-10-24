import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def verif_acc(pred, actual, claims, gi):
	verif=0
	for a,b,c,d in zip(pred,actual, claims, gi):
		c=np.argmax(c)
		if(d==1 and a==b):
			verif+=1
		if(d==0 and a!=c):
			verif+=1
	return verif/len(pred)

def roc(genuine, imposter, ROC):
	TPR=[]
	FPR=[]
	th=0
	while(th<=1):
		tp=len(genuine[genuine>=th])
		fp=len(imposter[imposter>=th])
		tn=len(imposter[imposter<th])
		fn=len(genuine[genuine<th])
		FPR.append(fp/(fp+tn))
		TPR.append(tp/(tp+fn))
		th+=0.001
	plt.figure(1)
	a,=plt.plot(FPR,TPR)
	ROC.append(a)
	return TPR,FPR

def analyse_helper(sess,model,X_test,Y_test,claims,gi,ROC):
	x=tf.placeholder('float', shape=X_test.shape)
	prediction=model.get_logits(x)
	scores=tf.nn.softmax(prediction)
	scores=sess.run(scores, feed_dict={x:X_test})
	genuine=[]
	imposter=[]
	for a,b in zip(Y_test,scores):
		a=np.argmax(a)
		genuine.append(b[a])
		imposter=imposter+list(b[:a])+list(b[(a+1):])
	genuine=np.array(genuine)
	imposter=np.array(imposter)
	TPR,FPR=roc(genuine,imposter,ROC)

	pred=model.model.predict(X_test)
	pred=np.argmax(pred, axis=1)
	actual=np.argmax(Y_test, axis=1)

	return verif_acc(pred,actual,claims,gi)    

def verification_report(sess,model,X_test,Y_test,claims,gi,adv=None,mit=None,save='./roc.jpg'):
	report={}
	ROC=[]
	leg=[]
	report["Verification Accuracy (before attack)"]=analyse_helper(sess,model,X_test,Y_test,claims,gi,ROC)
	leg.append("Before Attack")
	if(adv is not None):
		report["Verification Accuracy (after attack)"]=analyse_helper(sess,model,adv,Y_test,claims,gi,ROC)
		leg.append("After Attack")

	if(mit is not None):
		if type(mit) is np.ndarray:
			report["Verification Accuracy (after mitigation)"]=analyse_helper(sess,model,mit,Y_test,claims,gi,ROC)	
		else:
			report["Verification Accuracy (after mitigation)"]=analyse_helper(sess,mit,adv,Y_test,claims,gi,ROC)
		leg.append("After Mitigation")
		
	plt.figure(1)
	plt.xscale('log')
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("ROC curve")

	plt.legend(tuple(ROC), tuple(leg), loc='lower right')
	plt.savefig(save)
	plt.close()
	return report


def print_verification_report(sess,model,X_test,Y_test,claims,gi,adv=None,mit=None,save='./roc.jpg'):
	report=verification_report(sess,model,X_test,Y_test,claims,gi,adv,mit,save)
	print("\n")
	print("------------------------ VERIFICATION REPORT ------------------------")
	print("\n")

	for k,v in report.items():
		print(k+" : "+str(v*100)+"%")
	print("ROC Location : "+save)	

	print("\n")
	print("---------------------------------------------------------------------")
	print("\n")
	return report
