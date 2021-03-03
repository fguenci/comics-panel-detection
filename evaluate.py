from comics_dataset import ComicsDataset
from comics_evaluate import ComicsEvaluate
from comics_panel_detection.mrcnn.model import MaskRCNN
import tensorflow as tf

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from comics_panel_detection.mrcnn.visualize import display_instances


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# load the train dataset
train_set = ComicsDataset()
train_set.load_dataset('dataset', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# load the test dataset
test_set = ComicsDataset()
test_set.load_dataset('dataset', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# create config
cfg = ComicsEvaluate()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
""" model.load_weights('coco-model/mask_rcnn_comics_cfg_0010.h5', by_name=True)

# evaluate model on training dataset
train_mAP = train_set.evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = train_set.evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)
 """
#VEIFICA PREDIZIONE
# load model weights
model_path = 'coco-model/mask_rcnn_comics_cfg_0005.h5'
model.load_weights(model_path, by_name=True)
# plot predictions for train dataset
#train_set.plot_actual_vs_predicted(train_set, model, cfg, 1)
# plot predictions for test dataset
#test_set.plot_actual_vs_predicted(test_set, model, cfg, 1)


for path,dirs,files in os.walk('dataset\\test'):
    for f in files:
        # load photograph
        full_file_name = os.path.relpath(os.path.join(path,f))
        img = load_img(full_file_name)
        img = img_to_array(img)
        # make prediction
        results = model.detect([img], verbose=0)
        # get dictionary for first prediction
        r = results[0]
        # show photo with bounding boxes, masks, class labels and scores
        display_instances(img, r['rois'], r['masks'], r['class_ids'], train_set.class_names, r['scores'])
