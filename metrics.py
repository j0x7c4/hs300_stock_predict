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
        f1_map = {}
        for k, v in correct_map.items():
            p =1.0*v / predict_map[k] if k in predict_map else 0
            c = 1.0* v/ cls_label_map[k]
            precision_map[f"precision@{k}"] = p
            recall_map[f"recall@{k}"] = c
            f1_map[f"f1@{k}"] = 2*p*c /(p+c) if p+c > 0 else 0
        print(correct_map)
        print(cls_label_map)
        ret = np.average(list(precision_map.values()))
        info = precision_map
        info.update(recall_map)
        info.update(f1_map)
        return ret, info
