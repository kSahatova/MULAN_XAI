from maskrcnn.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import pprint
import random
import numpy as np
import logging

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn

from maskrcnn.config import cfg, merge_a_into_b, cfg_from_file
from maskrcnn.utils.comm import synchronize, get_rank
from maskrcnn.utils.logger import setup_logger
from maskrcnn.config import cfg

from time import time
import nibabel as nib
from tqdm import tqdm
import cv2
from openpyxl import load_workbook

from maskrcnn.data.datasets.load_ct_img import load_prep_img
from maskrcnn.structures.image_list import to_image_list
from maskrcnn.data.datasets.evaluation.DeepLesion.post_process import post_process_results
from maskrcnn.data.datasets.load_ct_img import windowing, windowing_rev
from maskrcnn.utils.draw import draw_results

from maskrcnn.data import make_data_loader, make_datasets
from maskrcnn.solver import make_optimizer
from maskrcnn.modeling.detector import build_detection_model
from maskrcnn.utils.checkpoint import DetectronCheckpointer
from maskrcnn.utils.comm import synchronize, get_rank
from maskrcnn.utils.miscellaneous import mkdir
from maskrcnn.config import cfg
from maskrcnn.engine.demo_process import import_tag_data, get_ims, load_preprocess_nifti, gen_output, print_msg_on_img


device = torch.device('cuda:0')

def check_configs():
    if cfg.MODE in ('train',):
        cfg.TEST.USE_SAVED_PRED_RES = 'none'
    elif cfg.MODE in ('vis',):
        cfg.TEST.EVAL_SEG_TAG_ON_GT = False
        cfg.LOG_IN_FILE = False
    elif cfg.MODE in ('demo', 'batch'):
        cfg.TEST.USE_SAVED_PRED_RES = 'none'
        cfg.TEST.EVAL_SEG_TAG_ON_GT = False

    scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
    assert scales == cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
    if cfg.MODEL.BACKBONE.FEATURE_UPSAMPLE:
        assert len(scales) == 1 and scales[0] == 1. / 2**(cfg.MODEL.BACKBONE.FEATURE_UPSAMPLE_LEVEL-1)
    anchor = cfg.MODEL.RPN.ANCHOR_STRIDE
    assert len(anchor) == 1 and anchor[0] == 1. / scales[0]

    if not cfg.MODEL.USE_3D_FUSION:
        assert cfg.INPUT.NUM_IMAGES_3DCE == 1
        assert cfg.MODEL.BACKBONE.FEATURE_FUSION_LEVELS == [False] * 3

    if cfg.GPU == '':
        import GPUtil
        deviceIDs = GPUtil.getAvailable(order='lowest', limit=1, maxMemory=.2)
        if len(deviceIDs) == 0:
            deviceIDs = GPUtil.getAvailable(order='lowest', limit=1, maxMemory=.9, maxLoad=1)
        cfg.GPU = str(deviceIDs[0])


def merge_test_config(test_config_file):
    cfg_new = cfg_from_file(test_config_file)
    # cfg.GPU = cfg_new.GPU
    cfg.TEST.TEST_SLICE_INTV_MM = cfg_new.TEST_SLICE_INTV_MM
    cfg.TEST.VISUALIZE.SCORE_THRESH = cfg_new.DETECTION_SCORE_THRESH
    cfg.TEST.VISUALIZE.DETECTIONS_PER_IMG = cfg_new.MAX_DETECTIONS_PER_IMG
    cfg.TEST.MIN_LYMPH_NODE_DIAM = cfg_new.MIN_LYMPH_NODE_DIAM_TO_SHOW
    cfg.TEST.MASK.THRESHOLD = cfg_new.MASK_THRESHOLD
    cfg.TEST.VISUALIZE.NMS = cfg_new.BBOX_NMS_OVERLAP
    cfg.INPUT.IMG_DO_CLIP = cfg_new.IMG_DO_CLIP
    cfg.TEST.TAGS_TO_KEEP = cfg_new.TAGS_TO_KEEP
    cfg.TEST.RESULT_FIELDS = cfg_new.RESULT_FIELDS

    return cfg_new

  
def build_model(config_file, test_config_file):
    cfg_new = cfg_from_file(config_file)
    merge_a_into_b(cfg_new, cfg)

    log_dir = cfg.LOGFILE_DIR
    logger = setup_logger("maskrcnn", log_dir, cfg.EXP_NAME, get_rank())

    if cfg.MODE in ('demo'):
        cfg_test = merge_test_config(test_config_file)
        logger.info(pprint.pformat(cfg_test))
    else:
        logger.info("Loaded configuration file {}".format(config_file))
        logger.info(pprint.pformat(cfg_new))
    check_configs()

    cfg.runtime_info.local_rank = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.runtime_info.distributed = num_gpus > 1
    if cfg.runtime_info.distributed:
        torch.cuda.set_device(cfg.runtime_info.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    logger.info("Using {} GPUs".format(num_gpus))

    #logger = logging.getLogger('maskrcnn.train')
    datasets = make_datasets('train')
    logger.info('building model ...')
    model = build_detection_model()  # some model parameters rely on initialization of the dataset
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    optimizer = None

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(model, optimizer, scheduler=None, save_dir=cfg.CHECKPT_DIR,
                                        prefix=cfg.EXP_NAME, save_to_disk=save_to_disk)

    if cfg.BEGIN_EPOCH == 0:
        if not cfg.MODEL.INIT_FROM_PRETRAIN:
            logger.info('No pretrained weights')
        else:
          # extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
          model.backbone.load_pretrained_weights()
    else:
        name = checkpointer.get_save_name(cfg.BEGIN_EPOCH, prefix=cfg.FINETUNE_FROM)
        extra_checkpoint_data = checkpointer.load(name)

    return model, checkpointer


