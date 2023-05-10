import torch
import torchvision


def importData():
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    print("import data from Caltech101")
    ROOT = ""
    Caltech101 = torchvision.datasets.Caltech101(root=ROOT, download='true')
    data_loader = torch.utils.data.DataLoader(Caltech101,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=8)
    print("save data in to file")
    return [Caltech101,data_loader]
