from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# Train options
__C.TRAIN = edict()
__C.TRAIN.OBJ_NAMES = ('sugarbeet', 'weed')
__C.TRAIN.IMGSZ = 512
__C.TRAIN.GRIDSZ = 16
__C.TRAIN.ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

# TEST options
__C.TEST = edict()
