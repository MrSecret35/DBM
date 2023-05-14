from torchvision.models import resnet50, ResNet50_Weights
import genericFunction as GF

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
ReteNeurale.avgpool.register_forward_hook(GF.get_features('avgpool',features))
ReteNeurale.layer3.register_forward_hook(GF.get_features('layer3',features))
#
#--------------------------------------------------