from comics_panel_detection.mrcnn.config import Config

# define the prediction configuration
class ComicsEvaluate(Config):
    # define the name of the configuration
    NAME = "comics_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 2
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1# calculate the mAP for a model on a given dataset
    USE_MINI_MASK = False