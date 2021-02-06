from comics_dataset import ComicsDataset
from comics_evaluate import ComicsEvaluate
from comics_panel_detection.mrcnn.model import MaskRCNN
import tensorflow as tf

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
#model.load_weights('coco-model/mask_rcnn_comics_cfg_0005.h5', by_name=True)

""" # evaluate model on training dataset
train_mAP = train_set.evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = train_set.evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP) """

# load model weights
model_path = 'coco-model/mask_rcnn_comics_cfg_0005.h5'
model.load_weights(model_path, by_name=True)
# plot predictions for train dataset
train_set.plot_actual_vs_predicted(train_set, model, cfg)
# plot predictions for test dataset
test_set.plot_actual_vs_predicted(test_set, model, cfg)