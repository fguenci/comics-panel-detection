# split into train and test set
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from numpy import expand_dims
from numpy import mean
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

# class that defines and loads the kangaroo dataset
class ComicsDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "Vignetta")
		self.add_class("dataset", 2, "Baloon")
		# define data locations
		images_dir = dataset_dir + '/image/'
		annotations_dir = dataset_dir + '/pascal-voc/'
		count = 0
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			count += 1
			# skip all images after 150 if we are building the train set
			if is_train and count >= 100:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and count < 100:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=count, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for obj in root.findall('.//object'):
			label = obj.find('name').text
			box = obj.find('bndbox')
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax, label]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index(box[4]))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

	def evaluate_model(self, dataset, model, cfg):
		APs = list()
		for image_id in dataset.image_ids:
			# load image, bounding boxes and masks for the image id
			image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id)
			# convert pixel values (e.g. center)
			scaled_image = mold_image(image, cfg)
			# convert image into one sample
			sample = expand_dims(scaled_image, 0)
			# make prediction
			yhat = model.detect(sample, verbose=0)
			# extract results for first sample
			r = yhat[0]
			# calculate statistics, including AP
			AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
			# store
			APs.append(AP)
		# calculate the mean AP across all images
		mAP = mean(APs)
		return mAP