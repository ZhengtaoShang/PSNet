import numpy as np

class Metric_polarity(object):
    def __init__(self) -> None:
        
        self.tppp_num= []
        self.fpnp_num= []
        self.fpup_num= []
        self.tnnn_num= []
        self.fnun_num= []
        self.fnpn_num= []
        self.tuuu_num= []
        self.fupu_num= []
        self.funu_num= []

        self.metrics = ( 'tppp', 'fpnp', 'fpup', 'fnpn', 'tnnn', 'fnun', 'fupu', 'funu', 'tuuu', 'prp', 'prn', 'pru', 'rep', 'ren', 'reu')
    
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
        self._cal_pr_re()
        self._round()
        metric_dict = {}
        for metric in self.metrics:
            metric_dict[metric] = self.get_metric(metric)
        return metric_dict

    
    def _cal_pr_re(self):
        tppp, fpnp, fpup, fnpn, tnnn, fnun, fupu, funu, tuuu = self.get_metric('tppp'), self.get_metric('fpnp'), self.get_metric('fpup'), self.get_metric('fnpn'), self.get_metric('tnnn'), self.get_metric('fnun'), self.get_metric('fupu'), self.get_metric('funu'), self.get_metric('tuuu')
        
        prp = 0 if (tppp==0 and fpnp==0) else tppp/(tppp+fpnp)
        prn = 0 if (tnnn==0 and fnpn==0) else tnnn/(tnnn+fnpn)
        pru = 0 if (tuuu==0 and fupu==0 and funu==0) else tuuu/(tuuu+fupu+funu)

        rep = 0 if (tppp==0 and fnpn==0 and fupu==0) else tppp/(tppp+fnpn+fupu)
        ren = 0 if (tnnn==0 and fpnp==0 and funu==0) else tnnn/(tnnn+fpnp+funu)
        reu = 0 if (tuuu==0 and fpup==0 and fnun==0) else tuuu/(tuuu+fpup+fnun)

        self._set_metric('prp', prp); self._set_metric('prn', prn); self._set_metric('pru', pru)
        self._set_metric('rep', rep); self._set_metric('ren', ren); self._set_metric('reu', reu); 


        
    def _round(self):
        for metric in self.metrics:
            value = self.get_metric(metric)
            if isinstance(value, np.float64) or isinstance(value, float):
                self._set_metric(metric, float(np.round(value, 5)))
        
        