from easydict import EasyDict as edict

__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

__C.POINTS_NUMBER             = 1

# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = 'datalab'
__C.TRAIN.POSTTIVE_SAMPLE     = 'train_have.tfrecords'
__C.TRAIN.NEGATIVE_SAMPLE     = 'train.tfrecords'

__C.TRAIN.BATCH_SIZE          = 8
__C.TRAIN.BATCH_NUMS          = 4000
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = [496,656] # 256 #[492,658]
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.EPOCHS              = 30
__C.TRAIN.RL                 = 1e-3



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = 'datalab2'
__C.TEST.BATCH_SIZE           = 4
__C.TEST.INPUT_SIZE           = [496,656] # [492,658]
__C.TEST.DATA_AUG             = False

