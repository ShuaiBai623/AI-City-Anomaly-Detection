import numpy as np
from .custom import CustomDataset
import json


class AnomalyDataset(CustomDataset):

    CLASSES = ('car')
    def load_annotations(self, ann_file):
       
        img_infos = []
        with open(ann_file, 'r') as f:
          for line in f:
              tmp_ann={}
              gt_bboxes = []
              gt_labels = []
              gt_bboxes_ignore = []
              data = json.loads(line)
              tmp_ann['filename'] = data["filename"]
              tmp_ann['width'] = 800#data["image_width"]
              tmp_ann['height'] = 410#data["image_height"]
              for box in data["instances"]:
                if box["is_ignored"]:
                  gt_bboxes_ignore.append(box['bbox'])
                else:
                  gt_bboxes.append(box["bbox"])
                  gt_labels.append(box["label"])
              if gt_bboxes:
                  gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
                  gt_labels = np.array(gt_labels, dtype=np.int64)
              else:
                  gt_bboxes = np.zeros((0, 4), dtype=np.float32)
                  gt_labels = np.array([], dtype=np.int64)

              if gt_bboxes_ignore:
                  gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
              else:
                  gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

              ann = dict(bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)
              tmp_ann['ann'] = ann
              img_infos.append(tmp_ann)


        return img_infos
