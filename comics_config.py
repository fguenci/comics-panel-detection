from comics_panel_detection.mrcnn.config import Config


class ComicsConfig(Config):
    # Give the configuration a recognizable name
    NAME = "comics_cfg"
    # Number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 2
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 170
    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2