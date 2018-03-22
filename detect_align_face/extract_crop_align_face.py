import warnings
warnings.filterwarnings('ignore')

from scipy import misc
import sys
import os
import random
from tqdm import tqdm
import argparse
import numpy as np
import cv2
import tensorflow as tf

import facenet
import detect_face

# define align function
from skimage import transform as trans
def align_and_crop(img, src, points):
	tform = trans.SimilarityTransform()
	pset_x = points[:5]
	pset_y = points[5:]
	dst = np.array(list(zip(pset_x, pset_y))).astype(np.float32).reshape(5,2)
	tform.estimate(dst,src)
	M = tform.params[:2,:]
	#M = cv2.estimateRigidTransform(dst.reshape(1,5,2), src.reshape(1,5,2), False)
	warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
	return warped

# define variables going to use
ROOT_DIR = os.getcwd()
DATA_PATH = os.path.join(ROOT_DIR, 'data')
MODEL_PATH = os.path.join(ROOT_DIR, 'model')
MTCNN_MODEL_PATH = os.path.join(MODEL_PATH, 'mtcnn')
IMG_IN_PATH = 'C:/Data1/CASIA-WebFace'
IMG_OUT_PATH = os.path.join(DATA_PATH, "CASIA-WebFace-Crop")

random_key = np.random.randint(0, high=99999)
bounding_boxes_filename = os.path.join(IMG_OUT_PATH, 'bounding_boxes_%05d.txt' % random_key)

if not os.path.exists(IMG_OUT_PATH):
	os.makedirs(IMG_OUT_PATH)

dataset = facenet.get_dataset(IMG_IN_PATH)
print('Total face identities: ', len(dataset))

print('Creating networks and loading parameters')

with tf.Graph().as_default():
	sess = tf.Session()
	with sess.as_default():
		pnet, rnet, onet = detect_face.create_mtcnn(sess, MTCNN_MODEL_PATH)

minsize = 20
threshold = [0.6, 0.7, 0.9]
factor = 0.85

image_size = (112, 96)
x_ = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299]
y_ = [51.6963, 51.5014, 71.7366, 92.3655, 92.2041]
src = np.array(list(zip(x_,y_))).astype(np.float32).reshape(5,2)

random_key = np.random.randint(0, high=99999)
bounding_boxes_filename = os.path.join(IMG_OUT_PATH, 'bounding_boxes_%05d.txt' % random_key)

# crop image


with open(bounding_boxes_filename, 'w') as text_file:
	nrof_images_total = 0
	nrof_successfully_aligned = 0
	
	for cls in tqdm(dataset):
		output_class_dir = os.path.join(IMG_OUT_PATH, cls.name)
		if not os.path.exists(output_class_dir):
			os.makedirs(output_class_dir)
		for image_path in cls.image_paths:
			nrof_images_total += 1
			filename = os.path.splitext(os.path.split(image_path)[1])[0] #img_name
			output_filename = os.path.join(output_class_dir, filename+'.png') #output_img_name
			
			if not os.path.exists(output_filename):
				try:
					img = cv2.imread(image_path)
				except (IOError, ValueError, IndexError) as e:
					errorMessage = '{}: {}'.format(image_path, e)
				else:
					if img.ndim < 2:
						print('Unable to align "%s"' % image_path)
						text_file.write('%s\n' % (output_filename))
						continue
					if img.ndim == 2:
						img = facenet.to_rgb(img)
						print('to_rgb data dimension: ', img.ndim)
					img = img[:,:,:3]
					bounding_boxes, points = detect_face.detect_face(img, minsize,pnet,rnet,onet,threshold,factor)
					nrof_faces = bounding_boxes.shape[0]
					if nrof_faces > 0:
						det = bounding_boxes[:,:4]
						img_size = np.asarray(img.shape)[:2]
						if nrof_faces > 1:
							bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
							img_center = img_size/2
							offsets = np.vstack([(det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0]])
							offset_dist_squared = np.sum(np.power(offsets,2.0),0)
							index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
							det = det[index,:]
							points = points[:,index]
						det = np.squeeze(det)
						out = align_and_crop(img, src, points)
						nrof_successfully_aligned += 1
						cv2.imwrite(output_filename, out)
						text_file.write('%s Success\n' % (output_filename))
					else:
						text_file.write('%s\n' % (output_filename))
			
'''			
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--model', type=str, default='')
	ROOT_DIR = os.getcwd()
	DATA_PATH = os.path.join(ROOT_DIR, 'data')
	MODEL_PATH = os.path.join(ROOT_DIR, 'model')
	MTCNN_MODEL_PATH = os.path.join(MODEL_PATH, 'mtcnn')
	IMG_IN_PATH = os.path.join(DATA_PATH, "CASIA-WebFace")
	IMG_OUT_PATH = os.path.join(DATA_PATH, "CASIA-WebFace-Crop")
'''

