# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:40:28 2017

@author: Zhdun
"""
class UnsupervisedModel:
    """
    简单无监督排序方法
    """
    def __init__(self,attrName=None):
        """
        attrName:比如(loc_A)
        """
        t=["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm", "lcom3","loc",
           "dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc"]

        self.attrIndex=t.index(attrName[0:-2])
        self.attrName=attrName
        
    def fit(self,X=None,y=None):
        pass
        
    def predict(self,X):
        sign=self.attrName[-1]
        y=list(X[:,self.attrIndex])
        if(sign=='A'):    
            y=[(lambda e: 1.0/e if e!=0 else float("inf"))(x) for x in y]
        return y
class LocRfc:
    """
    简单无监督排序方法
    """
    def __init__(self,sign="D"):
        """
        sign:D  or A
        

        """
        t=["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm", "lcom3","loc",
           "dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc"]

        self.index1=t.index("loc")
        self.index2=t.index("rfc")
        self.sign=sign
    def fit(self,X,y):
        pass
        
    def predict(self,X):
        y1=list(X[:,self.index1])
        y2=list(X[:,self.index2])
        y=[]   
        for i in range(len(y1)):
            y.append(y1[i]+y2[i])
        if(self.sign=='A'):    
            y=[(lambda e: 1.0/e if e!=0 else float("inf"))(x) for x in y]
        return y




