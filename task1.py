import torch
import torch.nn as nn

import genericFunction as GF

def createData(Caltech101,ReteNeurale,features):
    d = Extractor(1,Caltech101,ReteNeurale,features)
    # print(d)
    print(len(d[0]))
    print(len(d[1]))
    print(len(d[2]))
    print(d[3])


def Extractor(imgID,Caltech101,ReteNeurale,features):
    img, label = Caltech101[imgID]
    tensImg = GF.IMGtoTensor(img)
    prediction = ReteNeurale(tensImg).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()

    vector_last = ReteNeurale(tensImg).squeeze(0)

    temp_tensor_avgpool = torch.tensor(features['avgpool'].cpu().numpy().squeeze(0))
    tensor_avgpool = temp_tensor_avgpool.reshape(1, 1, -1)
    k = nn.AvgPool1d(2, stride=2)
    vector_avgpool = k(tensor_avgpool).flatten()

    tensor_layer3 = torch.tensor(features['layer3'].cpu().numpy().squeeze(0))
    m = nn.AvgPool2d(14, stride=1)
    vector_layer3 = m(tensor_layer3).flatten()

    return [vector_last, vector_avgpool, vector_layer3, class_id]
