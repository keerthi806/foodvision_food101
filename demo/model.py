import torch
import torchvision
from torch import nn

def create_effnetB2(num_classes:int=3, seed:int=42):
	effnetB2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
	effnetB2_transforms = effnetB2_weights.transforms()

	effnetB2 = torchvision.models.efficientnet_b2(weights=effnetB2_weights)

	for param in effnetB2.parameters():
		param.requires_grad = False

	torch.manual_seed(seed)

	effnetB2.classifier = nn.Sequential(
		nn.Dropout(p=0.3, inplace=True),
		nn.Linear(in_features=1408, out_features=num_classes)
	)

	return effnetB2, effnetB2_transforms
