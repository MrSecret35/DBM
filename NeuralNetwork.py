import torchvision
from torchvision import datasets, models, transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.models import resnet50, ResNet50_Weights

def get_features(name,features):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


#--------------------------------------------------
#
# definizione di una rete Neurale
#
features = {}
default_weights = ResNet50_Weights.DEFAULT
resnet50(weights=ResNet50_Weights.DEFAULT)
ReteNeurale = resnet50(weights=default_weights)
ReteNeurale.eval()
#utilizzato per prendere gli output dei layer specificati
ReteNeurale.avgpool.register_forward_hook(get_features('avgpool',features))
ReteNeurale.layer3.register_forward_hook(get_features('layer3',features))
#
#--------------------------------------------------

# prepare the function to preprocess images to be compatible with ResNet50
default_weights = ResNet50_Weights.DEFAULT
preprocess = default_weights.transforms()


def IMGtoTensor(img):
  if torchvision.transforms.functional.get_image_num_channels(img) != 1:
    proc_img = preprocess(img).unsqueeze(0)
    return proc_img
  else:
    print ("incompatiable image format -- try another one")