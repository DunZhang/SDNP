# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 13:48:13 2017

@author: Zhdun
"""
import pandas as pd
import numpy as np
class Data:
    """
    用来获取训练集和测试集
    """
    def __init__(self,paths,scenario,testSetName,trainingProjName):
        """
        Parameters:
        ----------
        paths:
            文件路径集合
            
        scenario:
            应用场景，Within-Version，Cross-Version，Cross-Project
            
        testSetName:
            测试集名称
            
        trainingProjName:
            训练集名称
        """
        self.paths=paths
        self.scenario=scenario
        self.testSetName=testSetName
        self.trainingProjName=trainingProjName
    def __getPaths(self):  
        """
        获取训练集路径和测试集路径集合
        Return:
        -------
        (训练集路径集合，测试集路径集合)
        """
        trainingSetsPaths,testSetsPaths=[],[]
        # In[]获取测试集路径
        for i in self.paths[self.testSetName.split("-")[0]]:
            if(self.testSetName in i):
                testSetsPaths.append(i)
                break
        # In[]获取训练集
        if(self.scenario=="Within-Version"):
            trainingSetsPaths=testSetsPaths
        elif(self.scenario=="Cross-Version"):
            for i in self.paths[self.testSetName.split("-")[0]]:
                if(self.testSetName in i):
                    break
                trainingSetsPaths.append(i)
        elif(self.scenario=="Cross-Project"):
            trainingSetsPaths=self.paths[self.trainingProjName]
        return (trainingSetsPaths,testSetsPaths)
    
    def __getDataSets(self,paths):
        data=[]
        for i in paths:
            data.append(np.array(pd.read_csv(i),dtype=float))
        data=np.vstack(data)
        return data[:,0:-1],data[:,-1]
    
    def getTrainingAndTestSet(self):
        """
        返回未经处理直接读取的训练集和测试集
        
        (x_train,y_train,x_test,y_test)
        """
        ps=self.__getPaths()
        x_train,y_train=self.__getDataSets(ps[0])
        x_test,y_test=self.__getDataSets(ps[1])
        return x_train,y_train,x_test,y_test
    
    
if (__name__=="__main__"):
    datasetsPaths={#Windows 下的路径
               "ant":["bugs\\ant-1.3.csv",
                       "bugs\\ant-1.4.csv",
                       "bugs\\ant-1.5.csv",
                       "bugs\\ant-1.6.csv",
                       "bugs\\ant-1.7.csv"],
               "camel":["bugs\\camel-1.0.csv",
                       "bugs\\camel-1.2.csv",
                       "bugs\\camel-1.4.csv",
                       "bugs\\camel-1.6.csv"],
               "jedit":["bugs\\jedit-3.2.csv",
                       "bugs\\jedit-4.0.csv",
                       "bugs\\jedit-4.1.csv",
                       "bugs\\jedit-4.2.csv"],
               "synapse":["bugs\\synapse-1.0.csv",
                       "bugs\\synapse-1.1.csv",
                       "bugs\\synapse-1.2.csv"],
               "ivy":["bugs\\ivy-1.1.csv",
                      "bugs\\ivy-1.4.csv",
                      "bugs\\ivy-2.0.csv"],
               "xalan":["bugs\\xalan-2.4.csv",
                        "bugs\\xalan-2.5.csv",
                        "bugs\\xalan-2.6.csv"],
                "xerces":["bugs\\xerces-1.2.csv",
                          "bugs\\xerces-1.3.csv"]
               }
    x_train,y_train,x_test,y_test=Data(paths=datasetsPaths, scenario="Within-Version", testSetName="ant-1.7", trainingProjName="camel").getTrainingAndTestSet()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    