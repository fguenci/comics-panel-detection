# split into train and test set
from os import error, listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from numpy import expand_dims
from numpy import mean
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import os
import fnmatch
from skimage import draw
from pprint import pprint
from xml.dom import minidom

# class that defines and loads the kangaroo dataset
class ComicsDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		
		def extract_polygon():
			# load and parse the file
			tree = ElementTree.parse(ann_path)
			# get the root of the document
			root = tree.getroot()
			try:
				if 'svg' not in root.tag:
					width, height, polygons = loadFromXml(root)
				else:
					width, height, polygons = loadFromSvg()
			except ValueError as error:
				print('File in errore: ' + ann_path)
				raise error

			return width, height, polygons

		def loadFromXml(root):
			polygons = []
			for obj in root.findall('.//object'):
				label = obj.find('name').text
				box = obj.find('bndbox')
				xmin = int(box.find('xmin').text)
				ymin = int(box.find('ymin').text)
				xmax = int(box.find('xmax').text)
				ymax = int(box.find('ymax').text)
				allX = [xmin, xmin, xmax, xmax]
				allY = [ymin, ymax, ymax, ymin]
				polygons.append({'all_points_x': allX, 'all_points_y': allY, 'label':label})
				if label == '7':
					print(minidom.parseString(ElementTree.tostring(root)).toprettyxml(indent="   "))
					raise ValueError("load_dataset - found label: '%s'" % label)
			# extract image dimensions
			width = int(root.find('.//size/width').text)
			height = int(root.find('.//size/height').text)
			return width, height, polygons

		def loadFromSvg():
			from xml.dom.minidom import parse

			polygons = []
			xmldoc = parse(ann_path)
			image = xmldoc.getElementsByTagName('image') 
			width = int(image[0].attributes['width'].value)
			height = int(image[0].attributes['height'].value)

			itemlist = xmldoc.getElementsByTagName('polygon') 
				
			for s in itemlist :
				label = s.parentNode.attributes['class'].value
				if label == 'Panel':
					label = 'Vignetta'
				elif label == 'Balloon':
					label = 'Baloon'
				else:
					continue
				xList = []
				yList = []				
				points = s.attributes['points'].value
				coords = points.split(' ')          
				#Compute bounding box
				for c in coords:
					pts = c.split(',')
					xList.append(int(pts[0]))
					yList.append(int(pts[1]))  
				polygons.append({'all_points_x': xList, 'all_points_y': yList, 'label':label})
			return width, height, polygons

		# define one class
		self.add_class("dataset", 1, "Vignetta")
		self.add_class("dataset", 2, "Baloon")
		# define data locations
		for path, dirs, files in os.walk(dataset_dir):
			num_files = len(fnmatch.filter(files,'*.xml'))
			count = 0
			for f in fnmatch.filter(files,'*.xml'):
				image_path = path.replace('groundtruth', 'image')
				image_id = os.path.relpath(os.path.join(image_path,f[:-3]+'jpg'))
				if not os.path.exists(image_id):
					image_id = image_id[:-3]+'png'
				if not os.path.exists(image_id):
					raise FileNotFoundError("load_dataset - No such file: '%s'" % image_id)
				ann_path = os.path.relpath(os.path.join(path,f))
				count += 1
				# skip all images after 150 if we are building the train set
				if is_train and count >= (num_files * 0.8):
					continue
				# skip all images before 150 if we are building the test/val set
				if not is_train and count < (num_files * 0.8):
					continue
				# add to dataset
				width, height, polygons = extract_polygon()
				self.add_image('dataset', 
				               image_id=image_id, 
							   path=image_id, 
							   annotation=ann_path,
							   width=width, 
							   height=height,
							   polygons=polygons)
    	
	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		print()
		# create one array for all masks, each on a different channel
		mask = zeros([info["height"], info["width"], len(info["polygons"])], dtype='uint8')
		# create masks
		class_ids = list()
		for i, p in enumerate(info["polygons"]):
			# Get indexes of pixels inside the polygon and set them to 1
			rr, cc = draw.polygon(p['all_points_y'], p['all_points_x'], mask.shape)
			try:
				class_ids.append(self.class_names.index(p['label']))
				mask[rr, cc, i] = 1
			except ValueError as error:
				pprint(p)
				pprint(class_ids)
				pprint(self.class_names)
				print('File: ' + info["annotation"])
				raise error
			except IndexError as error:
				pprint(p)
				pprint(class_ids)
				pprint(self.class_names)
				print('File: ' + info["annotation"])
				raise error

		return mask, asarray(class_ids, dtype='int32')

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
	
	# plot a number of photos with ground truth and predictions
	def plot_actual_vs_predicted(self, dataset, model, cfg, n_images=5):
		# load image and mask
		for i in range(n_images):
			info = self.image_info[i]['path']
			# load the image and mask
			image = dataset.load_image(i)
			mask, _ = dataset.load_mask(i)
			# convert pixel values (e.g. center)
			scaled_image = mold_image(image, cfg)
			# convert image into one sample
			sample = expand_dims(scaled_image, 0)
			# make prediction
			yhat = model.detect(sample, verbose=0)[0]
			pprint(yhat)
			# define subplot
			pyplot.subplot(n_images, 2, i*2+1)
			# plot raw pixel data
			pyplot.imshow(image)
			pyplot.title('Actual -> ' + info)
			# plot masks
			for j in range(mask.shape[2]):
				pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
			# get the context for drawing boxes
			pyplot.subplot(n_images, 2, i*2+2)
			# plot raw pixel data
			pyplot.imshow(image)
			pyplot.title('Predicted')
			ax = pyplot.gca()
			# plot each box
			for box in yhat['rois']:
				# get coordinates
				y1, x1, y2, x2 = box
				# calculate width and height of the box
				width, height = x2 - x1, y2 - y1
				# create the shape
				rect = Rectangle((x1, y1), width, height, fill=False, color='red')
				# draw the box
				ax.add_patch(rect)
		# show the figure
		pyplot.show()

""" 	def train(self, disable_gpu = False):
		
		if (disable_gpu):
			os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
		
		# train set
		train_set = ComicsDataset()
		train_set.load_dataset('dataset', is_train=True)
		train_set.prepare()
		print('Train: %d' % len(train_set.image_ids))
		# test/val set
		test_set = ComicsDataset()
		test_set.load_dataset('dataset', is_train=False)
		test_set.prepare()
		print('Test: %d' % len(test_set.image_ids))
		# prepare config
		config = ComicsConfig()
		config.display()
		# define the model
		model = MaskRCNN(mode='training', model_dir='./', config=config)
		# load weights (mscoco) and exclude the output layers
		model.load_weights('coco-model/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
		# train weights (output layers or 'heads')
		model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads') """