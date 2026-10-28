import numpy as np
import torch
from skimage import measure


class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, bins):
        # bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        # nclass :有几个类别 红外弱小目标检测只有一个类别
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)
        # self.reset()

    # 网络输入的结果和标签 计算两者之前的东西
    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            # score_thresh = (iBin + 0.0) / self.bins
            score_thresh = -30 + iBin * (255 / self.bins)
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)  # tp_rates = recall = TP/(TP+FN)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)  # fp_rates =  FP/(FP+TN)
        FP = self.fp_arr / (self.neg_arr + self.pos_arr)
        recall = self.tp_arr / (self.pos_arr + 0.001)  # recall = TP/(TP+FN)
        precision = self.tp_arr / (self.class_pos + 0.001)  # precision = TP/(TP+FP)

        return tp_rates, fp_rates, recall, precision, FP

    def reset(self):
        self.tp_arr = np.zeros([11])
        self.pos_arr = np.zeros([11])
        self.fp_arr = np.zeros([11])
        self.neg_arr = np.zeros([11])
        self.class_pos = np.zeros([11])


# class mIoU():

#     def __init__(self):
#         super(mIoU, self).__init__()
#         self.reset()

#     def update(self, preds, labels):
#         correct, labeled = batch_pix_accuracy(preds, labels)
#         inter, union = batch_intersection_union(preds, labels)
#         self.total_correct += correct  # 预测正确的像素数
#         self.total_label += labeled  # GT目标的像素数
#         self.total_inter += inter  # 交集
#         self.total_union += union  # 并集

#     def get(self):
#         pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
#         IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
#         mIoU = IoU.mean()
#         return float(pixAcc), mIoU

    # def reset(self):
    #     self.total_inter = 0
    #     self.total_union = 0
    #     self.total_correct = 0
    #     self.total_label = 0

class IoU():
    def __init__(self):
        super(IoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        return float(pixAcc), float(IoU)  # 直接返回IoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


class PD_FA():
    def __init__(self, ):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        predits = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64')

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)   # 目标总数  直接就搞GT的连通域个数
        self.image_area_total = []   # 图像中预测的区域列表
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        for i in range(len(coord_label)):   # image 与 label 之间 根据中心点 进行连通域的确定
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    self.image_area_match.append(area_image)

                    del coord_image[m]   # 匹配上一个之后就 清除一个
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match] # 在image里面 但是不在label里面
        self.dismatch_pixel += np.sum(self.dismatch)  # Fa 虚警
        self.all_pixel += size[0] * size[1]
        self.PD += len(self.distance_match)  # 如果中心点之间距离在3一下 就算Pd  所以Pd 是匹配上了的目标的个数

    def get(self):
        Final_FA = self.dismatch_pixel / self.all_pixel
        Final_PD = self.PD / self.target
        return Final_PD, float(Final_FA.cpu().detach().numpy())

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])

# ...existing code...

class nIoU():
    def __init__(self):
        super(nIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        inter, union = batch_intersection_union(preds, labels)
        self.total_inter += inter
        self.total_union += union

    def get(self):
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        # 归一化IoU，这里直接返回IoU均值，如有特殊归一化需求可自行修改
        nIoU = IoU.mean()
        return nIoU

    def reset(self):
        self.total_inter = 0
        self.total_union = 0

class F1Score():
    def __init__(self):
        super(F1Score, self).__init__()
        self.reset()

    def update(self, preds, labels):
        preds = (preds > 0).float()
        labels = (labels > 0).float()
        tp = (preds * labels).sum().item()
        fp = (preds * (1 - labels)).sum().item()
        fn = ((1 - preds) * labels).sum().item()
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def get(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0


def batch_pix_accuracy(output, target):
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()  # 将output 从 True Flase 转成 1 0
    pixel_labeled = (target > 0).float().sum()  # GF中 1的个数
    pixel_correct = (((predict == target).float()) * ((target > 0)).float()).sum()  # 预测对的个数
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
