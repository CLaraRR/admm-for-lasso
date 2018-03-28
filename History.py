#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 2018.3.27

@author:ningrun
'''
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class History:
    def __init__(self):
        self.objval = []
        self.r_norm = []
        self.s_norm = []
        self.eps_pri = []
        self.eps_dual = []

    def addObjval(self, objval):
        self.objval.append(objval)

    def getObjval(self):
        return self.objval

    def addR_norm(self, r_norm):
        self.r_norm.append(r_norm)

    def getR_norm(self):
        return self.r_norm

    def addS_norm(self, s_norm):
        self.s_norm.append(s_norm)

    def getS_norm(self):
        return self.s_norm

    def addEps_pri(self, eps_pri):
        self.eps_pri.append(eps_pri)

    def getEps_pri(self):
        return self.eps_pri

    def addEps_dual(self, eps_dual):
        self.eps_dual.append(eps_dual)

    def getEps_dual(self):
        return self.eps_dual
