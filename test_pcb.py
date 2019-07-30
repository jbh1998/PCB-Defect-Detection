from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import os

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

def union(au, bu):
	x = min(au[0], bu[0])
	y = min(au[1], bu[1])
	w = max(au[2], bu[2]) - x
	h = max(au[3], bu[3]) - y
	return x, y, w, h
	
def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0, 0, 0, 0
	return x, y, w, h
	
def cal_iou(a, b):
	# a and b should be (x1,y1,x2,y2)

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	i = intersection(a, b)
	u = union(a, b)

	area_i = i[2] * i[3]
	area_u = u[2] * u[3]
	return float(area_i) / float(area_u)

def non_max_suppression_fast_with_index(boxes, probs, overlap_thresh=0.9, max_boxes=300):
	# code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# sort the bounding boxes 
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the intersection

		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])

		# find the union
		xx1_un = np.minimum(x1[i], x1[idxs[:last]])
		yy1_un = np.minimum(y1[i], y1[idxs[:last]])
		xx2_un = np.maximum(x2[i], x2[idxs[:last]])
		yy2_un = np.maximum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		ww_int = np.maximum(0, xx2_int - xx1_int)
		hh_int = np.maximum(0, yy2_int - yy1_int)

		ww_un = np.maximum(0, xx2_un - xx1_un)
		hh_un = np.maximum(0, yy2_un - yy1_un)

		# compute the ratio of overlap
		overlap = (ww_int*hh_int)/(ww_un*hh_un + 1e-9)

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlap_thresh)[0])))

		if len(pick) >= max_boxes:
			break
	# print 'nms input length:', probs.shape, 'nms output length:', len(idxs)
	# return only the bounding boxes that were picked using the integer data type
	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	return boxes, probs, pick	
	
classes = ['background',
           'open', 'short', 'mousebite', 'spur',
           'copper', 'pin-hole']

# Set the image size.
img_height = 640
img_width = 640
img_channels = 1
n_classes = 6
confidence_threshold = 0.9
target_class = None
batch_size = 16  # batch size when testing
 
# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.04, 0.08, 0.12, 0.24],
                    aspect_ratios_global=[0.5, 1.0, 2.0],
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=True,
                    steps=None,
                    offsets=None,
                    clip_boxes=False,
                    variances=[1.0, 1.0, 1.0, 1.0] ,
                    normalize_coords=True,
                    subtract_mean=127.5,
                    divide_by_stddev=127.5,
					top_k=200)

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = '/opt/Data/tsl/DeepPCB/ssd-PCB-GPP-MaxPooling/ssd7_epoch-490_loss-0.2872.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

cnt = 0
test_batch = []
temp_batch = []
img_names = []
gt_names = []
imgs = []
batch_idx = 0
img_path = './images/'
gt_path = './images/'
result_path = './results/'
test_txt = '/opt/Data/tsl/dataset/PCBData/test.txt'
# test_txt = './11060.txt'
test_img_paths = []
test_gt_paths = []

# warm up
# model.predict([np.zeros(shape=(1, img_height, img_width, 1)), np.zeros(shape=(1, img_height, img_width, 1))])

test_file = open(test_txt, 'r')
for img_anno_pair in test_file:
	# print img_anno_pair
	img_path, anno_path = img_anno_pair.split(' ')
	img_path = img_path.strip()
	anno_path = anno_path.strip().replace('\r', '')
	test_img_path = img_path[:-4] + "_test.jpg"
	test_img_paths.append(test_img_path)
	test_gt_paths.append(anno_path)

test_size = len(test_img_paths)

for idx, img_name in enumerate(test_img_paths):
	
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	if img_name[-8:-4] != 'test':
		continue
	test_img_name = img_name
	temp_img_name = img_name[0:-8]+"temp.jpg"
	
	img_names.append(img_name)
	gt_names.append(test_gt_paths[idx])
	
	test_img = cv2.imread(test_img_name, 0)
	temp_img = cv2.imread(temp_img_name, 0)

	imgs.append(test_img)
	X_test = np.expand_dims(test_img, 2)
	X_temp = np.expand_dims(temp_img, 2)
	X_test = np.expand_dims(X_test, 0)
	X_temp = np.expand_dims(X_temp, 0)
	
	test_batch.append(X_test)
	temp_batch.append(X_temp)
	batch_idx += 1
	
	if batch_idx < batch_size:
		# print cnt, batch_idx, test_size
		if cnt + batch_idx < test_size:
			continue
		else:
			batch_size = batch_idx
	print 'current batch size:', batch_size
	batch_idx = 0
	X_test_batch = np.concatenate(test_batch, axis=0)
	X_temp_batch = np.concatenate(temp_batch, axis=0)
	
	test_batch = []
	temp_batch = []
	
	st = time.time()
	y_pred = model.predict([X_test_batch, X_temp_batch])
	
	y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
	
	for pred_idx, box in enumerate(y_pred_thresh):
		nms_out = non_max_suppression_fast_with_index(box[:, -4:], box[:, 1], overlap_thresh=0.05, max_boxes=1000)
		if len(nms_out) < 3:
			y_pred_thresh[pred_idx] = []
		else:
			_, _, new_idx = nms_out
			y_pred_thresh[pred_idx] = box[new_idx, :]
	print 'ellapse time:', time.time() - st

	
	# visualize and save results
	for bc_id in range(batch_size):
		result_txt = open(os.path.join(result_path, "res_{}.txt".format(cnt)), 'w')
		
		for box in y_pred_thresh[bc_id]:
			(x1, y1, x2, y2) = box[-4:]
			tag = int(box[0])
			score = box[1]
			if target_class is None or classes[tag]==target_class:
				result_txt.write("{},{},{},{},{},{},{},{},{},{}\n".format(int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2), score, classes[tag]))
				
			
		result_txt.close()
		gt_dets = []
		img_scaled = cv2.cvtColor(imgs[bc_id], cv2.COLOR_GRAY2BGR)
		
		if os.path.exists(gt_names[bc_id]):
			gt_file = open(gt_names[bc_id], 'r')
			
			for lines in gt_file:
				lines = lines.strip()
				xmin, ymin, xmax, ymax, tag = lines.split(' ')
				gt_key = int(tag)
				gt_dets.append((gt_key, (int(xmin), int(ymin), int(xmax), int(ymax))))
			gt_file.close()
		
		gt_dets_visited = np.zeros(len(gt_dets), np.uint8)
		for box in gt_dets:
			color = (0, 255, 0)
			x1, y1, x2, y2 = box[1]
			cv2.rectangle(img_scaled,(x1, y1), (x2, y2), color, 2)
			textLabel = '{}'.format(classes[int(box[0])])
			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.5,1)
			textOrg = (x1, y1-10)
			cv2.rectangle(img_scaled, (textOrg[0] - 5, textOrg[1]+baseLine), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img_scaled, (textOrg[0] - 5,textOrg[1]+baseLine), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img_scaled, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
		for box in y_pred_thresh[bc_id]:
			best_matched_gt_id = -1
			max_iou = 0
			for id_gt, gt_det in enumerate(gt_dets):
				
				iou = cal_iou(gt_det[1], box[-4:])
				if iou>0.01:
					if iou>max_iou:
						best_matched_gt_id = id_gt
						max_iou = iou
						
			if best_matched_gt_id == -1:  ## wujian, red
				color = (0, 0, 255)
			else:
				if gt_dets[best_matched_gt_id][0] == box[0]:  # correct, green
					color = (0, 255, 0)
				else:
					color = (255, 0, 0)  # incorrect, blue
				gt_dets_visited[best_matched_gt_id] = 1
			x1, y1, x2, y2 = box[-4:].astype(np.int32)
			cv2.rectangle(img_scaled,(x1, y1), (x2, y2), color, 2)	
			# draw the detected proposals
			textLabel = '{}: {:.2f}'.format(classes[int(box[0])],box[1])
			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.5,1)
			textOrg = (x1, y1-2)
        
			cv2.rectangle(img_scaled, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img_scaled, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img_scaled, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
		
		for gt_id in range(len(gt_dets)):
			if gt_dets_visited[gt_id] == 0: ## loujian
				gt_det = gt_dets[gt_id]
				x1, y1, x2, y2 = gt_det[1]
				textLabel = '{}'.format(classes[gt_det[0]])
				(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.5,1)
				textOrg = (x1, y1-0)
        
				cv2.rectangle(img_scaled,(x1, y1), (x2, y2), (0, 0, 255), 2)	
				cv2.rectangle(img_scaled, (textOrg[0] - 5, textOrg[1]+baseLine), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
				cv2.rectangle(img_scaled, (textOrg[0] - 5,textOrg[1]+baseLine), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
				cv2.putText(img_scaled, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
		
		
		img_name = img_names[bc_id][-17:]
		try:
			gt_file = open(gt_names[bc_id], 'r')
			gt_out = open(os.path.join(result_path, "gt_{}.txt".format(cnt)), 'w')
			for lines in gt_file:
				lines = lines.strip()
				xmin, ymin, xmax, ymax, tag = lines.split(' ')
				tag = int(tag)
				if target_class is None or classes[tag]==target_class:
					gt_out.write("{},{},{},{},{},{},{},{},{}\n".format(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, classes[int(tag)]))
			gt_file.close()
			gt_out.close()
		except:
			pass
		cv2.imwrite(os.path.join(result_path, img_name), img_scaled)
		cnt += 1
	
	imgs = []
	img_names = []
	gt_names = []

		
