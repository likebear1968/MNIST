import numpy as np
from abc import ABCMeta, abstractmethod
import itertools
from enum import Enum

class Factory():
    class MTYPE(Enum): R2=1; MSE=2; MAE=3; ACC=4; PRC=5; RCL=6; F1=7; MTR=8;
    @staticmethod
    def create(mtype, lbl=[0,1]):
        if mtype == Factory.MTYPE.R2:  return R2()
        if mtype == Factory.MTYPE.MSE: return MSE()
        if mtype == Factory.MTYPE.MAE: return MAE()
        if mtype == Factory.MTYPE.ACC: return Accuracy(lbl)
        if mtype == Factory.MTYPE.PRC: return Precision(lbl)
        if mtype == Factory.MTYPE.RCL: return Recall(lbl)
        if mtype == Factory.MTYPE.F1:  return F1(lbl)
        if mtype == Factory.MTYPE.MTR: return Confusion_matrix(lbl)
        return None

class Metrics(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, obs, prd):
        '''
        評価値を取得する。
        obs：実測値
        prd：予測値
        戻り値：評価値
        '''
        pass

class R2(Metrics):
    def evaluate(self, obs, prd):
        return 1 - np.sum((obs - prd) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

class MSE(Metrics):
    def evaluate(self, obs, prd):
        return np.average((obs - prd) ** 2, axis=0)

class MAE(Metrics):
    def evaluate(self, obs, prd):
        return np.average(np.abs(prd - obs), axis=0)

class Confusion_matrix(Metrics):
    def __init__(self, lbl=[0,1]):
        self.lbl = lbl
        self.s = len(self.lbl)
        self.delta = 1e-7

    def evaluate(self, obs, prd):
        mtr = self.get_matrix(obs, prd)
        return self.accuracy(None, None, mtr), self.precision(None, None, mtr), self.recall(None, None, mtr), self.f1(None, None, mtr)

    def get_matrix(self, obs, prd):
        if isinstance(obs, list): obs = np.array(obs)
        if isinstance(prd, list): prd = np.array(prd)
        prd = self.normalize(prd)
        obs = obs.reshape(prd.shape[0], -1)
        if prd.shape != obs.shape:
            obs = self.normalize(obs)
        matrix = []
        for a, b in itertools.product(self.lbl, repeat=2):
            matrix.append(len(np.where((obs == a) & (prd == b))[0]))
        return np.array(matrix).reshape(self.s, self.s)

    def normalize(self, val):
        if val.ndim == 1: val = val.reshape(-1, 1)
        if val.shape[1] == 1:
            val = np.argmax(np.c_[1 - val, val], axis=1)
        else:
            val = np.argmax(val, axis=1)
        return val.reshape(-1, 1)
    
    def accuracy(self, obs, prd, mtr=None):
        if mtr is None: mtr = self.get_matrix(obs, prd)
        return np.sum(np.diag(mtr) + self.delta) / (np.sum(mtr) + self.delta)

    def precision(self, obs, prd, mtr=None):
        if mtr is None: mtr = self.get_matrix(obs, prd)
        return np.average((np.diag(mtr) + self.delta) / (np.sum(mtr, axis=0) + self.delta))

    def recall(self, obs, prd, mtr=None):
        if mtr is None: mtr = self.get_matrix(obs, prd)
        return np.average((np.diag(mtr) + self.delta) / (np.sum(mtr, axis=1) + self.delta))

    def f1(self, obs, prd, mtr=None):
        if mtr is None: mtr = self.get_matrix(obs, prd)
        r = self.recall(None, None, mtr)
        p = self.precision(None, None, mtr)
        return 2 * (r * p) / (r + p)

class Accuracy(Confusion_matrix):
    def __init__(self, lbl=[0,1]):
        super().__init__(lbl)

    def evaluate(self, obs, prd):
        return super().accuracy(obs, prd)

class Precision(Confusion_matrix):
    def __init__(self, lbl=[0,1]):
        super().__init__(lbl)

    def evaluate(self, obs, prd):
        return super().precision(obs, prd)

class Recall(Confusion_matrix):
    def __init__(self, lbl=[0,1]):
        super().__init__(lbl)

    def evaluate(self, obs, prd):
        return super().recall(obs, prd)

class F1(Confusion_matrix):
    def __init__(self, lbl=[0,1]):
        super().__init__(lbl)

    def evaluate(self, obs, prd):
        return super().f1(obs, prd)
