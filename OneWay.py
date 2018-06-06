# -*- coding: utf-8 -*-
from UnsupervisedModel import UnsupervisedModel
from Evaluate import Eva
class OneWay:
    def __init__(self,sign="D",metric="FPA"):
        self.bestAttrIndex=None
        self.bestAttrName=None
        self.sign=sign
        self.metric=metric
    def fit(self,X,y):
        t=["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm", "lcom3","loc",
           "dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc"]
        bestValue=float("-inf")
        for i in t:
            un=UnsupervisedModel(attrName=i+"_"+self.sign)
            tValue=Eva(un,X,y,self.metric)
            if(tValue>bestValue):
                self.bestAttrName=i+"_"+self.sign
                bestValue=tValue
        self.bestAttrIndex=t.index(self.bestAttrName[0:-2])
    def predict(self,X):
        y=list(X[:,self.bestAttrIndex])
        if(self.sign=="A"):
            y=[(lambda e: 1.0/e if e!=0 else float("inf"))(x) for x in y]
        return y  