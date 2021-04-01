import os
import cv2
import random
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
predictor = DefaultPredictor(cfg)
colors = []
for _ in range(100):
    colors.append([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
for i in range(13,100):
    path = "/VisualGroup/share/zjf/data/huadong_data/rope/imgs/160100-160100-011-05-600.avi/{:06d}.jpg".format(i)
    im = cv2.imread(path)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    pred_boxes = outputs["instances"]._fields["pred_boxes"].tensor.detach().cpu().numpy()
    scores = outputs["instances"]._fields["scores"].detach().cpu().numpy()
    pred_classes = outputs["instances"]._fields["pred_classes"]
    masks = outputs["instances"].pred_masks.detach().cpu().numpy()
    for idx, (bbox, s, mask) in enumerate(zip(pred_boxes, scores, masks)):
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
        cv2.putText(im, "{:.4f}".format(s), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),2)

    all_mask = np.zeros(im.shape, dtype=np.uint8)
    for idx, mask in enumerate(masks):
        tmp = np.ones(im.shape, dtype=np.uint8)
        mask = np.expand_dims(mask, 2)
        mask = mask.repeat(3,axis=2)
        tmp = tmp * mask
        tmp = tmp * colors[idx]
        all_mask += tmp.astype(np.uint8)
    im = 0.5 * im.astype(np.uint8) +  0.5 * all_mask.astype(np.uint8)
    cv2.imwrite("im.jpg", im)
    import time
    time.sleep(100000)