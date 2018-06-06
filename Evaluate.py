# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 13:10:59 2017

@author: Administrator
"""
from copy import deepcopy
import numpy as np
def Kendall(model,x,y):
    """
    计算Kendall,越接近0代表两组变量相关性越小，模型越接近于垃圾随机模型
    Parameter:
    ----------
    
    model:训练好的模型
    
    testSet：测试集，要有loc和bug
    
    """ 
    #获取预测结果
    Y=list(y)     
    X = list(model.predict(x))
    n,n1,n2=len(X),0,0
    
    for i in range(n):
        for j in range(i+1,n):
            if(X[i]>X[j] and Y[i]>Y[j]):
                n1=n1+1
            elif(X[i]<X[j] and Y[i]<Y[j]):
                n1=n1+1
            elif(X[i]>X[j] and Y[i]<Y[j]):
                n2=n2+1
            elif(X[i]<X[j] and Y[i]>Y[j]):
                n2=n2+1

    return (2*(n1-n2))/(n*(n-1))

def FPA(model,x,y):
    """
    计算FPA，必须是预测bug数目的模型
    Parameter:
    ----------
    
    model:训练好的模型
    
    """ 
    #获取预测结果

    bug=list(y)
    k=len(bug);
    n=sum(bug);    
    predResults = list(model.predict(x))
    t=list(zip(bug,predResults))
    t.sort(key = lambda arg : arg[1])#按预测结果升序排序
    fpa=0
    for m in range(1,k+1):
        tvalue=0
        for i in range(k-m,k):
            tvalue=tvalue+t[int(i)][0]
        fpa=fpa+(tvalue/n)
    return fpa/k;

def Acc20(model,x,y,locIndex=10):
    """
    计算ACC20
    Parameter:
    ----------
    
    model:训练好的模型
    
    x:测试集的x
    
    y:测试集的y
    
    locIndex:loc列索引
    
    """            
    bug=list(y)
    loc=list(x[:,locIndex])
    loc20=int(sum(loc)*0.2)
    predResults = list(model.predict(x))    
    t=list(zip(bug,predResults,loc))
    t.sort(key = lambda arg : arg[1],reverse=True)#按预测结果降序排序    
#    t.sort(key = lambda arg : 0 if arg[2]==0 else arg[1]/arg[2],reverse=True)  

    locInspected=0
    bugInspected=0
    for i in t:
        if(locInspected>=loc20):
            break
        locInspected+=i[2]
        bugInspected+=i[0]
    return bugInspected/sum(bug)
def AccM20(model,x,y,locIndex=10):
    """
    Reacll@20
    Parameter:
    ----------
    
    model:训练好的模型
    
    x:测试集的x
    
    y:测试集的y
    
    locIndex:loc列索引
    
    """            
    bug=list(y)
    loc=list(x[:,locIndex])
    m20=int(len(x)*0.2)
    predResults = list(model.predict(x))    
    t=list(zip(bug,predResults,loc))
    t.sort(key = lambda arg : arg[1],reverse=True)#按预测结果降序排序    
#    t.sort(key = lambda arg : 0 if arg[2]==0 else arg[1]/arg[2],reverse=True)  

    mInspected=0
    bugInspected=0
    for i in t:
        if(mInspected>=m20):
            break
        mInspected+=1
        bugInspected+=i[0]
    return bugInspected/sum(bug)
def sizeBased(model,x,y,locIndex=10):
    """
    计算ACC20
    Parameter:
    ----------
    
    model:训练好的模型
    
    x:测试集的x
    
    y:测试集的y
    
    locIndex:loc列索引
    
    """            
    bug=list(y)
    loc=list(x[:,locIndex])
    predResults = list(model.predict(x))    
    t=list(zip(bug,predResults,loc))
    bugInspected=0
    for i in t:
        if(i[2]>=114):
            bugInspected+=i[0]
    return bugInspected/sum(bug)
def Eva(model,x,y,metric,locIndex=10):
    if(metric=="Kendall"):
        return Kendall(model,x,y)
    elif(metric=="FPA"):
        return FPA(model,x,y)
    elif(metric=="Acc20"):
        return Acc20(model,x,y,locIndex)
    elif(metric=="sizeBased"):
        return sizeBased(model,x,y,locIndex)
    elif(metric=="AccM20"):
        return AccM20(model,x,y,locIndex)




        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


