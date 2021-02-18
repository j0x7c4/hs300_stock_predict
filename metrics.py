# encoding:utf-8
import numpy as np

class Metrics(object):
    @staticmethod
    def mAP(preds, labels):
        cls_label_map = {}
        correct_map = {}
        predict_map = {}
        for label in labels:
            if label not in cls_label_map:
                cls_label_map[label] = 0
                correct_map[label] = 0
            cls_label_map[label] += 1
        for i, pred in enumerate(preds):
            label = labels[i]
            if pred not in predict_map:
                predict_map[pred] = 0
            predict_map[pred] += 1
            if pred == label:
                correct_map[label] += 1
        precision_map = {}
        recall_map = {}
        for k, v in correct_map.items():
            precision_map[f"precision@{k}"] = 1.0*v / predict_map[k] if k in predict_map else 0
            recall_map[f"recall@{k}"] = 1.0* v/ cls_label_map[k]
        print(correct_map)
        print(cls_label_map)
        ret = np.average(list(precision_map.values()))
        info = precision_map
        info.update(recall_map)
        return ret, info
