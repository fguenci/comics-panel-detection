from comics_dataset import ComicsDataset
from comics_config import ComicsConfig
from mrcnn.model import MaskRCNN
import tensorflow as tf
import os


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
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')