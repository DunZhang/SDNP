# -*- coding: utf-8 -*-

"""
@author: Zhdun
"""
import random
from SMOTEND import SMOTEND
import pandas as pd
import numpy as np
from DE import DE
from sklearn.model_selection import StratifiedKFold
from BuildingModel import buildModel
import Evaluate as E
class SMOTENDDE:
    def __init__(self,metric,modelName,DRange,F=0.7,CR=0.3,PopulationSize=30,Lives=8):
        self.metric,self.modelName,self.DRange,self.F,self.CR,self.PopulationSize,self.Lives=metric,modelName,DRange,F,CR,PopulationSize,Lives
    def transform(self,x,y):
        """
        用DE优化SMOTEND,针对特定数据集  模型和评估指标的优化
        
        Parameters:
        -------------------
        Drange,F,CR,PopulationSize,Live:
            DE的相关参数
            
        x,y:
            用于评估参数的数据集
            
        metric:
            评估指标
            
        modelName：
            用的模型
            
        times:
            验证时计算的次数 
            
        Return:
        -------
            (newX,newY,bestPars(array(1*d))) 
        """
        metric,modelName,DRange,F,CR,PopulationSize,Lives= self.metric,self.modelName,self.DRange,self.F,self.CR,self.PopulationSize,self.Lives
        #Setep1 获取差分进化后SMOTER参数
        de=DE(fitness=SMOTENDDE_Fitness, D=3, DRange=DRange, F=F, CR=CR, PopulationSize=PopulationSize, 
              Lives=Lives)
        paras,_= de.evolution(otherPars=(x,y,modelName,metric))
        #Step2 返回最后一代参数
        return paras    
def SMOTENDDE_Fitness(targetPars,otherPars):
    """
    SMOTENDDE的适应度函数
    
    Parameters:
    -------------------    
        targetPars:
            array(1*3),分别是k,m,r
        
        otherPars:
            tuple,(x,y,modelName,metric,times)
     
        Return:
        ---------
        the value of metric
    """
    k,m,r=targetPars[0,0],targetPars[0,1],targetPars[0,2]
    x,y,modelName,metric=otherPars
    smo=SMOTEND(k,m,r)
#    print("AAAAAAAAAAAAAAAAAAAAA")
    q=CrossValDE(modelName=modelName, X=x, y=y, metric=metric, procsessor=smo, cv=3,
               times=1, random_state=0)
#    print("BBBBBBBBBBBBBBBBBBBB",q)
    return q
def CrossValDE(modelName, X, y,metric,procsessor=None,cv=3,times=1,random_state=0):
    """
    优化DE时用的交叉验证
    """
    res=[]
    yt=[]
    for i in y:
        if(i==0):
            yt.append(0)
        else:
            yt.append(1)
    for t in range(times):
        skf=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state+t)
        indices=list(skf.split(X=X,y=yt))   
#        print (" ",time.strftime("%M:%S"))
        for k in indices:
            x_train,y_train,x_test,y_test=X[k[0]],y[k[0]],X[k[1]],y[k[1]]
            
            if(procsessor is not None):        
#                t=procsessor.transform(x_train,y_train)
#                print("t",len(t))
#                if(len(t)==3):
#                    qqq=1
#                    qqq+=1
                x_train,y_train=procsessor.transform(x_train,y_train)
            estimator=buildModel(x_train,y_train,modelName)
            res.append(E.Eva(estimator,x_test,y_test,metric))    
#        print (" ",time.strftime("%M:%S"))            
    res=np.array(res)
    return res.mean()    
    
if(__name__=='__main__'):
    data=pd.read_csv("bugs/ant-1.7.csv")
    data=np.array(data,dtype=float)
    x,y=data[:,0:-1],data[:,-1]
#    print (" ",time.strftime("%M:%S"))
    q=SMOTENDDE(metric="Kendall",modelName="EALR",times=1
                ,DRange=[list(range(1,21)),[0,1,2,3,4,5,6],(1,5)],
                F=0.7,CR=0.3,PopulationSize=30,Lives=2)
    xn,yn,bestParas=q.transform(x,y)
#    print (" ",time.strftime("%M:%S"))
            
        
  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        