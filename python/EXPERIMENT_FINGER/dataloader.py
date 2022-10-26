# %%
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os 
import json
import random

class JsonDataset(Dataset):
    def __init__(self,dataPath,dataScale=1):
        with open(dataPath,'r') as RawData:
            rawData = json.load(RawData)
        rawData = rawData['data']
        label, input = [],[]
        for rawDatum in rawData:
            input.append(rawDatum['actuation'])
            label.append(rawDatum['position'])
            
        label = np.array(label)
        input = np.array(input)
        self.label, self.input =torch.tensor(dataScale*label).to(torch.float), torch.tensor(input).to(torch.float)
        
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,idx):
        return self.input[idx].numpy(), self.label[idx].numpy()

    def GetIOScale(self,device):
        (inputVar,inputMean),(outputVar,outputMean) = torch.var_mean(self.input,0), torch.var_mean(self.label.reshape(-1,3), dim=0)
        inputVar,inputMean,outputVar,outputMean = inputVar.to(device),inputMean.to(device),outputVar.to(device),outputMean.to(device)
        
        # Adjust small variance #
        if (inputVar<1e-9).any():
            print("Warning: inputVar is less than 1e-9... Fixing it to 1")
            inputVar[inputVar<1e-9]   =  1
        if (outputVar<1e-9).any():
            print("Warning: outputVar is less than 1e-9... Fixing it to 1")
            outputVar[outputVar<1e-9] =  1

        inputSigma, outputSigma = torch.sqrt(inputVar), torch.sqrt(outputVar)
        IOScale = {'IScale':[inputMean,inputSigma], 'OScale':[outputMean,outputSigma]}
        
        return IOScale

    def GetDataDimension(self):
        inputDim = self.input.size()[-1]
        outputDim = self.label.size()[1]
        assert self.label.size()[-1] == 3
        return (inputDim,outputDim)

class JsonDataloader(DataLoader):
    def __init__(self,data_path, n_workers,batch, shuffle = True,dataScale=1000):
        self.dataset = JsonDataset(data_path,dataScale)
        super().__init__(self.dataset, batch_size=batch, shuffle=shuffle, num_workers=n_workers)
        
def get_FINGER_Dataset(BASEDIR):
    totalDataset   = JsonDataset(str(BASEDIR / 'finger_dataset_0904.json'), dataScale=1)
    trainDataset   = JsonDataset(str(BASEDIR / 'finger_dataset_0904.json'), dataScale=1)
    valDataset     = JsonDataset(str(BASEDIR / 'finger_dataset_0904.json'), dataScale=1)
    testDataset    = JsonDataset(str(BASEDIR / 'finger_dataset_0904.json'), dataScale=1)
    extTestDataset = JsonDataset(str(BASEDIR / 'finger_dataset_0904.json'), dataScale=1)
    
    label = totalDataset.label.unsqueeze(1)
    label_mean = torch.mean(label,dim=0); label_mean[:,-1] = 0
    label = label - label_mean
    
    temp = label[:,:,0].clone()
    label[:,:,0] = -label[:,:,1]
    label[:,:,1] = temp
    

    label = label + torch.tensor([10,10,.0])/1000
    
    input = totalDataset.input
    input = input[:,[0,2,1,3]]
    
    SortIdx = torch.argsort(torch.norm(input, p=2, dim=-1))
    label = label[SortIdx]; input = input[SortIdx]

    MUL = torch.tensor(
        [[1,0.5,0.25,0.5],
         [0.5,1,0.5,0.25],
         [0.25,0.5,1,0.5],
         [0.5,0.25,0.5,1]
         ])
    input = input @ MUL
    
    
    ## 해야할것1. 기존 train,val,test,extTest를 유지한채로 data를 내보낼것 -> 단 예전처럼 dict보다는 그냥 array로 내보내자. 코드를 덜 바꿀수 있어 통일성이 있음.
    ## actuation을 scale을 기존이랑 맞춰줘야함. ~0.5 정도
    input = input/16_000
    
    
    # args 대신 수기로 넣어줌
    InterpolateDatasetRatio = 0.3
    DATA_FRACTION           = 1
    
    
    Interpolate_label = label[:int(len(label)*InterpolateDatasetRatio)]
    Extrapolate_label = label[int(len(label)*InterpolateDatasetRatio):]
    
    Interpolate_input = input[:int(len(input)*InterpolateDatasetRatio)]
    Extrapolate_input = input[int(len(input)*InterpolateDatasetRatio):]
    
    trainDataset.label   = Interpolate_label[:int(len(Interpolate_label)*0.8 * DATA_FRACTION)]
    valDataset.label     = Interpolate_label[int(len(Interpolate_label)*0.8):int(len(Interpolate_label)*0.9)]
    testDataset.label    = Interpolate_label[int(len(Interpolate_label)*0.9):]
    extTestDataset.label = Extrapolate_label
    
    trainDataset.input   = Interpolate_input[:int(len(Interpolate_input)*0.8 * DATA_FRACTION)]
    valDataset.input     = Interpolate_input[int(len(Interpolate_input)*0.8):int(len(Interpolate_label)*0.9)]
    testDataset.input    = Interpolate_input[int(len(Interpolate_input)*0.9):]
    extTestDataset.input = Extrapolate_input

    return (trainDataset,valDataset,testDataset, extTestDataset)

def find_most_extrem(dataset):
    batchidxs = []
    for i in range(4):
        maxidx = torch.argmax(dataset.input[:,i])
        batchidxs.append(maxidx)
        _,label = dataset.__getitem__(maxidx)
        print("Furthest position in dimension ", i+1," \n", label)    
    
    return batchidxs
        
