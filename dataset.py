import json
import numpy as np

from os.path import join
from glob import glob
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

def get_rope_dicts():
    labelme_format_data_path = "/zhuangjf/gitae/project/data/rope/poly_labels"
    json_files = glob(join(labelme_format_data_path, "*.json"))
    dataset_dicts = []
    for idx, json_file in enumerate(json_files):
        with open(json_file, "r") as f:
            context = json.load(f)
        filename = context["imagePath"].replace("../", "/zhuangjf/gitae/project/data/rope/")
        height, width = context["imageHeight"], context["imageWidth"]
        record = {}
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        annos = context["shapes"]
        objs = []
        for anno in annos:
            px, py = [], []
            anno = anno["points"]
            for ann in anno:
                px.append(ann[0])
                py.append(ann[1])
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts