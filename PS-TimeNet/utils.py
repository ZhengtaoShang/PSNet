#%%
import torch
import logging
from sklearn.metrics import f1_score
import math  
import numpy as np
from collections import deque

#%%
def get_logger(name, log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    file_handler = logging.FileHandler("{}/{}_log.txt".format(log_dir, name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    formatter_io = logging.Formatter('%(message)s')
    terminal_handler = logging.StreamHandler()
    terminal_handler.setLevel(logging.INFO)
    terminal_handler.setFormatter(formatter_io)
    logger.addHandler(terminal_handler)

    return logger

#%%
def save_checkpoint(save_path, model, optim=None, epoch=None, best_loss=math.inf ):
    save_dict = {
        "model":model.state_dict(),
        "optim":None,
        "epoch":epoch,
        "best_loss":best_loss}
    if optim:
        save_dict['optim'] = optim.state_dict()

    torch.save(save_dict, save_path)


def load_checkpoint(save_path):
    checkpoint = torch.load(save_path)
    model = checkpoint['model']
    optim = checkpoint["optim"]
    epoch = checkpoint["epoch"]
    loss = checkpoint['best_loss']
    return model, optim, epoch, loss


def _trigger_onset(x, thres1=0.4, thres2=0.4): ################################为什么选择两个阈值
    """
    Calculate trigger on and off times.
    Given thres1 and thres2 calculate trigger on and off times from characteristic function.
    """
    ind1 = np.where(x > thres1)[0]
    if len(ind1) == 0:
        return []
    ind2 = np.where(x > thres2)[0]
    on = deque([ind1[0]])
    of = deque([-1])
    
    # determine the indices where x falls below off-threshold
    ind2_ = np.empty_like(ind2, dtype=bool)
    ind2_[:-1] = np.diff(ind2) > 1
    ind2_[-1] = True                    # last occurence is missed by the diff, add it manually
    of.extend(ind2[ind2_].tolist()) # select the value corresponding to the 'True'
    on.extend(ind1[np.where(np.diff(ind1) > 1)[0] + 1].tolist())

    pick = []
    while on[-1] > of[0]:
        while on[0] <= of[0]:
            on.popleft()
        while of[0] < on[0]:
            of.popleft()
        pick.append([on[0], of[0]])
    if len(pick) <= 1:
        return np.array(pick, dtype=np.int64)    
    
    # merge the detection that are too close
    merge_pick = []
    cur_pick = pick[0]
    for p in pick[1:]:
        if p[0] - cur_pick[1] < 5:
            cur_pick = [cur_pick[0], p[1]]
        else:
            merge_pick.append(cur_pick)
            cur_pick = p 
    merge_pick.append(cur_pick)
    return np.array(merge_pick, dtype=np.int64)


def _detect_peaks_cor(x, samples, threshold=0.5):

    """
    ----------
    Detect peaks in data based on their amplitude and other features.
    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    ind = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]  # 找到概率分布从负值变为正值的点
    
    if not ind.size:
        return ind
    # first and last values of x cannot be peaks
    if ind[0] == 0:
        ind = ind[1:]
    if ind[-1] == x.size-1:
        ind = ind[:-1] 
    # remove peaks < threshold
    if ind.size:
        ind = ind[x[ind] >= threshold] # 找到SS概率大与阈值的极大值点
    ind, correlation = _filter_peaks_cor(x, ind, samples)
    return ind, correlation


def _filter_peaks_cor(x, peaks, samples, duaration=100):
    if peaks.size == 0:
        correlation = 0
        return peaks, correlation
    if peaks.size == 1:
        peak_point = peaks[0]
        correlation = _claculate_pick_ref_cor(x, peak_point, samples, label_half_width=20)
        return peaks, correlation


    if peaks.size > 1:
        new_peaks = []
        cur_peak = peaks[0]
        for peak in peaks[1:]:
            if x[peak] > x[cur_peak]:
                cur_peak = peak
        new_peaks.append(cur_peak) # 保存最大的概率

        peak_point = new_peaks[0]
        correlation = _claculate_pick_ref_cor(x, peak_point, samples, label_half_width=20)

        return np.array(new_peaks), correlation


def _claculate_pick_ref_cor(x, peak_point, samples, label_half_width=20):
    HW = label_half_width
    sigma = 10
    label = np.zeros((samples),dtype=np.float32)
    predict = np.zeros((samples),dtype=np.float32)
    # SS_label
    if (peak_point-HW > 0) and (peak_point+HW < samples):  
        label[int(peak_point-HW):int(peak_point+HW)] = np.exp(-(np.arange(-HW,HW))**2/(2*(sigma)**2))
        predict[int(peak_point-HW):int(peak_point+HW)] = x[int(peak_point-HW):int(peak_point+HW)]
        label = label[int(peak_point-HW):int(peak_point+HW)]
        predict = predict[int(peak_point-HW):int(peak_point+HW)]

    elif (peak_point+HW) >= samples:
        label[int(peak_point-HW):samples] = np.exp(-(np.arange(-HW, samples-peak_point))**2/(2*(sigma)**2))
        predict[int(peak_point-HW):samples] = x[int(peak_point-HW):samples]
        label = label[int(peak_point-HW):samples]
        predict = predict[int(peak_point-HW):samples]

    elif (peak_point-HW) < samples:
        label[0:int(peak_point+HW)] = np.exp(-(np.arange(-peak_point, HW))**2/(2*(sigma)**2))
        predict[0:int(peak_point+HW)] = x[0:int(peak_point+HW)]
        label = label[0:int(peak_point+HW)]
        predict = predict[0:int(peak_point+HW)]

    correlation = np.corrcoef(predict,label)[0,1]
    return correlation


def picker_cor(SS, d, samples, thresholds = [0.9, 0.9, 0.5]):

    """ 
    Performs detection and picking.
    Parameters: 
    SS: 1D array, S arrival probabilities;
    d: 1D array, Detection probabilities. 
    """               

    detection = _trigger_onset(d, thres1=thresholds[0], thres2=thresholds[1])
    SS_arr, correlation = _detect_peaks_cor(SS, samples, threshold=thresholds[2])
    # print(detection)
    # print(SS_arr)

    SS_PICKS = []
    EVENTS = []

    if len(SS_arr) > 0:
        for i in range(0,len(SS_arr)): 
            SS_cur = SS_arr[i]      
            SS_prob = np.round(SS[int(SS_cur)], 3) 
            SS_PICKS.append([SS_cur, SS_prob])
            
    if len(detection) > 0:
        for begin,end in detection:
            D_prob = np.round(np.mean(d[begin:end]),3)
            EVENTS.append([begin, end, D_prob])          

    matches = []        
    for (bg,ed,D_prob) in EVENTS:
        matches.append({"detection":[bg, ed, D_prob]})

    for (Ss, S_val) in SS_PICKS:
        best_ss, best_ssv = Ss, S_val
        matches.append({"ss": [best_ss, best_ssv]})

    return matches, correlation



class Metric(object):
    def __init__(self, error_flag = False) -> None:
        self.error_flag = error_flag
        self.metrics = ('tp', 'tn', 'fp', 'fn', 'pr', 're', 'f1')
        if self.error_flag:
            self.errors = []
            self.errors_num = []
            self.bigerrors = []
            self.bigerrors_num = []
            self.fpnum = []
            self.fnnum = []
            self.tpnum = []
            self.tnnum = []

            self.metrics = ('tp', 'tn', 'fp', 'fn', 'pr', 're', 'f1', 'error_mean', 'error_std')

        for metric in self.metrics:
            self._set_metric(metric, 0)
  
    def get_metric(self, metric):
        return getattr(self, metric)
    
    def _set_metric(self, metric, value):
        setattr(self, metric, value)
    
    def update_metric(self, metric, value = 1):
        new_value = self.get_metric(metric) + value
        setattr(self, metric, new_value)
        
    def metric_dict(self):
        self.done()
        metric_dict = {}
        for metric in self.metrics:
            metric_dict[metric] = self.get_metric(metric)
        return metric_dict

    def done(self):
        self._cal_pr_re_f1()
        if self.error_flag:
            self._cal_error_mean_std()
        self._round()
    
    def _cal_pr_re_f1(self):
        tp, fp, fn = self.get_metric('tp'), self.get_metric('fp'), self.get_metric('fn')
            
        pr = 0 if (not tp and not fp) else tp / (tp + fp)
        re = 0 if (not tp and not fn) else tp / (tp + fn)
        f1 = 0 if (not pr and not re) else 2 * pr * re / (pr + re)
        self._set_metric('pr', pr)
        self._set_metric('re', re)
        self._set_metric('f1', f1)

        
    def _round(self):
        for metric in self.metrics:
            value = self.get_metric(metric)
            if isinstance(value, np.float64) or isinstance(value, float):
                self._set_metric(metric, float(np.round(value, 5)))
        
        
    def _cal_error_mean_std(self):
        error_mean = np.mean(np.array(self.errors)) if len(self.errors) else 0
        error_std = np.std(np.array(self.errors)) if len(self.errors) else 0

        self._set_metric('error_mean', error_mean)
        self._set_metric('error_std', error_std)
 