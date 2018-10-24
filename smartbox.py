import tensorflow as tf
import numpy as np
import keras
import pickle
from PIL import Image
from Model import GeneralModel
import argparse
from Datasets import Datasets
from Attacks.fgsm import FGSM
from Attacks.l2 import L2
from Attacks.mifgsm import MIFGSM
from Attacks.pgd import PGD
from Detection.AdaptiveNoiseReduction import AdaptiveNoiseReduction
from Detection.ConvFiltDetect import ConvFiltDetect
from Detection.LearnArtifacts import LearnArtifactsModel
from Detection.PCADetect import PCADetect
from Mitigation.AdversarialTraining import AdversarialTraining
from Mitigation.AutoEncoderTest import AutoEncoder
from Mitigation.GaussianBlur import GaussianBlur
from Mitigation.Randomization import Randomization
from Facial.identification import print_identification_report
from Facial.verification import print_verification_report

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parent_dataset=parser.add_argument_group('Dataset Information')
group = parent_dataset.add_mutually_exclusive_group(required=True)
group.add_argument("-nd","--new_dataset", action='store_true', help="Use a new custom dataset")
group.add_argument("-lda","--load_dataset", action='store_true', help="Use a dataset used before")
parent_dataset.add_argument("-dl","--dataset_location", type=str, default=None, help="Location of the new dataset (if --new_dataset is selected). Location from where the dataset has to be loaded (if --load_dataset is selected)")
parent_dataset.add_argument("-sdi","--save_dataset_info", type=str, default=None,help='File to save the newly loaded dataset information (if --new_dataset is selected)')
parent_dataset.add_argument("-trsp","--train_split", type=float, default=0.8,help='Training split')
parent_dataset.add_argument("-vasp","--validation_split", type=float, default=0.1,help='Validation split')
parent_dataset.add_argument("-tesp","--test_split", type=float, default=0.1,help='Testing split')



images=parser.add_argument_group('Images Information')
images.add_argument("-hei","--height", type=int, default=50, help="Height of the images")
images.add_argument("-wid","--width", type=int, default=50, help="Width of the images")
images.add_argument("-ch","--channels", type=int,choices=[1,3], default=1, help="1 for grayscale, 3 for RGB")
images.add_argument("-cmi","--clip_min", type=float, default=0.0, help="Minimum input component value")
images.add_argument("-cma","--clip_max", type=float, default=1.0, help="Maximum input component value")

modelinfo=parser.add_argument_group('Model Information')
modelinfo.add_argument("-e","--epochs", type=int, default=100, help="Number of epochs")
modelinfo.add_argument("-tbs","--training_batch_size", type=int, default=100, help="Batch Size during training")
modelinfo.add_argument("-s","--save_model", type=str,default=None, help="File to save the trained model in")
modelinfo.add_argument("-l","--load_model", type=str,default=None, help="Load Saved Model")
modelinfo.add_argument("-lr","--learning_rate", type=float, default=0.01, help="Learning Rate")
modelinfo.add_argument("-mom","--momentum", type=float, default=0.9, help="Momentum")


# parser.add_argument("-cmi","--clip_min", type=float, default=-0.5, help="Minimum input component value")
# parser.add_argument("-cma","--clip_max", type=float, default=0.5, help="Maximum input component value")

attackinfo=parser.add_argument_group('Attack Algorithm and Parameters')
attackinfo.add_argument("-a","--attack", type=str, choices=["L2", "FGSM", "PGD", "MOMENTUM"], help=("Attack to run"))
attackinfo.add_argument("-ep","--eps", type=float, default=0.1, help="Attack step size (use with FGSM, MOMENTUM or PGD)")
attackinfo.add_argument("-ana", "--attack_analysis", action='store_true',default=True, help="Print analysis of attacks")
attackinfo.add_argument("-sai","--save_attacked_images", action='store_true', default=True, help="Save attacked images?")
attackinfo.add_argument("-bsa","--attack_batch_size", type=int, default=100, help="Number of images to attack at a time (use with L2)")
attackinfo.add_argument("-epi","--eps_iter", type=float, default=0.06, help="Step size for attack iteration (use with MOMENTUM or PGD)")
attackinfo.add_argument("-df","--decay_factor", type=float, default=1, help="Decay Factor (use with MOMENTUM)")
attackinfo.add_argument("-iter","--iterations", type=int, default=10, help="Iterations to run attack for")


detectinfo=parser.add_argument_group('Detection Algorithm Information and Parameters')
detectinfo.add_argument("-dalg","--detect_algo", type=str, choices=["AdaptiveNoise", "PCA", "ArtifactLearning","ConvFilter"],default=None, help="Detection algorithm to run")
detectinfo.add_argument("-and", "--detection_analysis", action='store_true',default=True, help="Print analysis after detection")
detectinfo.add_argument("-mpl","--max_pooling_layers", type=int, default=3, help="Maximum number of pooling layers (use with ConvFilter)")
detectinfo.add_argument("-bsd","--artifact_batch_size", type=int, default=100, help="Number of samples to run at a time (use with ArtifactLearning)")
detectinfo.add_argument("-ed","--artifact_epochs", type=int, default=100, help="Number of epochs (use with ArtifactLearning)")
detectinfo.add_argument("-sd","--save_artifact_model", type=str,default=None, help="File to save the trained model in (use with ArtifactLearning)")
detectinfo.add_argument("-ld","--load_artifact_model", type=str,default=None, help="Load Saved Model (use with ArtifactLearning)")


mitigateinfo=parser.add_argument_group('Mitigation Algorithm Information and Parameters')
mitigateinfo.add_argument("-malg","--mitigate_algo", type=str, choices=["AdaptiveNoise", "AutoEncoder", "GaussianBlur", "Randomization", "AdversarialTraining"],default=None, help="Mitigation algorithm to run")
mitigateinfo.add_argument("-anm", "--mitigation_analysis", action='store_true',default=True, help="Print analysis after mitigation")
mitigateinfo.add_argument("-smi","--save_mitigated_images", action='store_true',default=True, help="Save mitigated images?")
mitigateinfo.add_argument("-bsm","--advtrae_batch_size", type=int, default=100, help="Number of samples to run at a time (use with AdversarialTraining or AutoEncoder)")
mitigateinfo.add_argument("-em","--advtrae_epochs", type=int, default=100, help="Number of epochs (use with AdversarialTraining or AutoEncoder)")
mitigateinfo.add_argument("-sm","--save_advtrae_model", type=str,default=None, help="File to save the trained model in (use with AdversarialTraining or AutoEncoder)")
mitigateinfo.add_argument("-lm","--load_advtrae_model", type=str,default=None, help="Load Saved Model (use with AdversarialTraining or AutoEncoder)")

# parser.add_argument("-ser", "--serial", action='store_true', help="Use images detected by Detection algo to feed in the Mitigation algo")

parser.add_argument("-ide", "--identification", action='store_true', help="Perform identification")
parser.add_argument("-ver", "--verification", action='store_true', help="Perform verification")

args = parser.parse_args()

if(args.save_attacked_images):
	args.attack_analysis=True

if(args.save_mitigated_images):
	args.mitigation_analysis=True

with tf.Session() as sess:
	height=args.height
	width=args.width
	channels=args.channels
	minval=args.clip_min
	maxval=args.clip_max
	x=tf.placeholder('float', shape=(None,height,width,channels))
	

	if(args.load_dataset):
		d=Datasets(height,width,channels,restore=args.dataset_location)

	else:
		d=Datasets(height,width,channels)
		d.load_dataset(args.dataset_location,train_split=args.train_split,valid_split=args.validation_split,test_split=args.test_split,minval=minval,maxval=maxval,save=args.save_dataset_info)


	d.get_info()
	training_data=(d.images_train,d.labels_train,d.images_valid,d.labels_valid)
	model=GeneralModel(height,width,d.classes,channels,args.load_model)

	if(args.load_model==None):
		model.train(training_data,args.epochs,args.training_batch_size,learning_rate=args.learning_rate, momentum=args.momentum ,modeldir=args.save_model)

	if(args.save_attacked_images):
		if(args.attack=="L2"):
			attack=L2(sess,model,save="./",batch_size=args.attack_batch_size,minval=minval,maxval=maxval)
		elif(args.attack=="FGSM"):
			attack=FGSM(sess, model, eps=args.eps,save="./",minval=minval,maxval=maxval)
		elif(args.attack=="MIFGSM"):
			attack=MIFGSM(sess, model, eps=args.eps,decay_factor=args.decay_factor,iterations=args.iterations,step=args.eps_iter,save="./",minval=minval,maxval=maxval)
		else:
			attack=PGD(sess, model, eps=args.eps,step=args.eps_iter,iterations=args.iterations,save="./",minval=minval,maxval=maxval)

	else:
		if(args.attack=="L2"):
			attack=L2(sess,model,save=None,batch_size=args.attack_batch_size,minval=minval,maxval=maxval)
		elif(args.attack=="FGSM"):
			attack=FGSM(sess, model, eps=args.eps,save=None,minval=minval,maxval=maxval)
		elif(args.attack=="MIFGSM"):
			attack=MIFGSM(sess, model, eps=args.eps,decay_factor=args.decay_factor,iterations=args.iterations,step=args.eps_iter,save=None,minval=minval,maxval=maxval)
		else:
			attack=PGD(sess, model, eps=args.eps,step=args.eps_iter,iterations=args.iterations,save=None,minval=minval,maxval=maxval)


	if(args.attack=="L2"):
		adv=attack.attack(d.images_test,d.targets)
	else:
		adv=attack.attack(d.images_test, d.labels_test)

	if(args.attack_analysis):
		attack.print_analysis_results(d.images_test, d.labels_test)

	if(args.detect_algo=="AdaptiveNoise"):
		det=AdaptiveNoiseReduction(sess, model, height, width, channels, minval, maxval)
	elif(args.detect_algo=="PCA"):
		det=PCADetect(sess, model, height, width, attack, trainX=d.images_train, trainY=d.labels_train, channels=channels, minval=minval, maxval=maxval)
	elif(args.detect_algo=="ConvFilter"):
		det=ConvFiltDetect(sess, model, height, width, attack, trainX=d.images_train, trainY=d.labels_train, maxpoollayers=args.max_pooling_layers, channels=channels, minval=minval, maxval=maxval)
	else:
		det=LearnArtifactsModel(sess, model, height, width, attack, trainX=d.images_train, trainY=d.labels_train, batch_size=args.artifact_batch_size, epochs=args.artifact_epochs,  channels=channels, minval=minval, maxval=maxval, restore=args.load_artifact_model,save=args.save_artifact_model)

	newx, newy = det.prepare_data(d.images_test, adv)
	detected=det.detect(newx)
	if(args.detection_analysis):
		det.print_analysis_results(newy)

	if(args.save_mitigated_images):
		if(args.mitigate_algo=="AdaptiveNoise"):
			mit=AdaptiveNoiseReduction(sess, model, height, width, channels, minval, maxval,save="./")
		elif(args.mitigate_algo=="AutoEncoder"):
			mit=AutoEncoder(sess, model, height, width, attack, trainX=d.images_train, trainY=d.labels_train, batch_size=args.advtrae_batch_size, epochs=args.advtrae_epochs,  channels=channels, minval=minval, maxval=maxval, restore=args.load_advtrae_model,save="./",modelsave=args.save_advtrae_model)
		elif(args.mitigate_algo=="GaussianBlur"):
			mit=GaussianBlur(sess,model,height, width,channels=channels, minval=minval, maxval=maxval, save="./")
		elif(args.mitigate_algo=="Randomization"):
			mit=Randomization(sess, model, height, width, channels=channels, minval=minval, maxval=maxval, save="./")
		else:
			mit=AdversarialTraining(sess, model, height, width, trainX=d.images_train, trainY=d.labels_train,attack=attack,channels=channels, minval=minval, maxval=maxval, save="./",batch_size=args.advtrae_batch_size, epochs=args.advtrae_epochs,restore=args.load_advtrae_model,modelsave=args.save_advtrae_model)
	else:
		if(args.mitigate_algo=="AdaptiveNoise"):
			mit=AdaptiveNoiseReduction(sess, model, height, width, channels, minval, maxval,save=None)
		elif(args.mitigate_algo=="AutoEncoder"):
			mit=AutoEncoder(sess, model, height, width, attack, trainX=d.images_train, trainY=d.labels_train, batch_size=args.advtrae_batch_size, epochs=args.advtrae_epochs,  channels=channels, minval=minval, maxval=maxval, restore=args.load_advtrae_model,save=None,modelsave=args.save_advtrae_model)
		elif(args.mitigate_algo=="GaussianBlur"):
			mit=GaussianBlur(sess,model,height, width,channels=channels, minval=minval, maxval=maxval, save=None)
		elif(args.mitigate_algo=="Randomization"):
			mit=Randomization(sess, model, height, width, channels=channels, minval=minval, maxval=maxval, save=None)
		else:
			mit=AdversarialTraining(sess, model, height, width, trainX=d.images_train, trainY=d.labels_train,attack=attack,channels=channels, minval=minval, maxval=maxval, save=None,batch_size=args.advtrae_batch_size, epochs=args.advtrae_epochs,restore=args.load_advtrae_model,modelsave=args.save_advtrae_model)

	mitigated = mit.mitigate(adv)
	if(args.mitigation_analysis):
		mit.print_analysis_results(X_test=adv, Y_test=d.labels_test)

	if(args.identification):
		rm=print_identification_report(sess,model,d.classes,d.images_test,d.labels_test,adv=adv,mit=mitigated)

	if(args.verification):
		rd=print_verification_report(sess,model,d.images_test,d.labels_test,d.targets,d.genuine_imposter,adv=adv,mit=mitigated)





