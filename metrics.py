# encoding:utf-8
import numpy as np

class Metrics(object):
    @staticmethod
    def mAP(preds, labels):
        cls_label_map = {}
        correct_map = {}
        for label in labels:
            if label not in cls_label_map:
                cls_label_map[label] = 0
                correct_map[label] = 0
            cls_label_map[label] += 1
        for i, pred in enumerate(preds):
            label = labels[i]
            if pred == label:
                correct_map[label] += 1
        precisions = []
        for k, v in correct_map.items():
            precisions.append(correct_map[k]*1.0 / cls_label_map[k])
        return np.average(precisions)
