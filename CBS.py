# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
class CBS:
    def __init__(self,attrName="loc_D",logis=None):
        """
        第二阶段按哪个属性排序
        """
        t=["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm", "lcom3","loc",
           "dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc"]

        self.attrIndex=t.index(attrName[0:-2])
        self.attrName=attrName
        self.logis=logis
    def fit(self,X,y):
        if(self.logis is None):
            yt=[(lambda e: 0 if e==0 else 1)(x) for x in y] 
            self.logis=LogisticRegression().fit(X,yt)
        else:
            print("WRONG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WRONG")
    
    def predict(self,X):
        yPred=list(self.logis.predict(X))
        for i in range(len(X)):
            if(yPred[i]==1):
                e=X[i,self.attrIndex]
                if(self.attrName[-1]=="D"):                    
                    yPred[i]=e
                else:
                    if(e!=0):
                        yPred[i]=1.0/e
                    else:
                        yPred[i]=float("inf")
        return yPred