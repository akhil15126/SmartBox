import os
import cv2



def save_images_for_comparison(name, actual, before, before_out, after, after_out, location,targets=None):
	folder=os.path.join(location, name)
	if not os.path.exists(folder):
		os.makedirs(folder)
	o_folder=os.path.join(folder, "before")
	a_folder=os.path.join(folder, "after")
	if not os.path.exists(o_folder):
		os.makedirs(o_folder)
	if not os.path.exists(a_folder):
		os.makedirs(a_folder)

	count=1
	# print(len(original))
	# print(len(original_out))
	if(targets is None):
		for a,b,c,d,e in zip(before, after,before_out,after_out,actual):
			a=a*255
			b=b*255
			label=str(count)+"_Predicted"+str(c)+"_Actual"+str(e)+"_"+str(c==e)+".jpg"
			cv2.imwrite(os.path.join(o_folder, label), a)
			label=str(count)+"_Predicted"+str(d)+"_Original"+str(c)+"_Actual"+str(e)+"_"+str(d==e)+".jpg"
			cv2.imwrite(os.path.join(a_folder, label), b)
			count+=1		
	else:
		for a,b,c,d,e,f in zip(before, after,before_out,after_out,actual,targets):
			a=a*255
			b=b*255
			label=str(count)+"_Predicted"+str(c)+"_Actual"+str(e)+"_"+str(c==e)+".jpg"
			cv2.imwrite(os.path.join(o_folder, label), a)
			label=str(count)+"_Predicted"+str(d)+"_Target"+str(f)+"_Original"+str(c)+"_Actual"+str(e)+"_"+str(d==e)+".jpg"
			cv2.imwrite(os.path.join(a_folder, label), b)
			count+=1



	








def save_images(images, location):
	pass