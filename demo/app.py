import gradio as gr
import torch
import os
from model import create_effnetB2
from timeit import default_timer as timer
from typing import Tuple, Dict
import spaces

with open("class_names.txt", "r") as f:
	class_names = [class_name.strip() for class_name in f.readlines()]

effnetB2, effnetB2_transforms = create_effnetB2(num_classes=101)

effnetB2.load_state_dict(
	torch.load(
		f="09_effnetB2_feature_extractor_food101_20_percent.pth",
		# map_location=torch.device("cpu")
	),
)

@spaces.GPU
def predict(img) -> Tuple[Dict, float]:
	start_time = timer()

	img = effnetB2_transforms(img).unsqueeze(0)

	effnetB2.eval()
	with torch.inference_mode():
		pred_probs = torch.softmax(effnetB2(img), dim=1).squeeze(0)

	pred_labels_and_scores = {class_names[i]: score.item() for i, score in enumerate(pred_probs)}

	pred_time = timer() - start_time

	return pred_labels_and_scores, pred_time

title = "Foodvision (Food101)"
description = "An [EfficientNetB2](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b2.html) feature extractor to classify food images of 101 classes from `FOOD101` dataset."
examples_list = [["examples/" + example] for example in os.listdir("examples")]

examples_list = [["examples/" + example_path] for example_path in os.listdir("examples")]

demo = gr.Interface(
	fn=predict,
	inputs=gr.Image(type="pil"),
	outputs=[
		gr.Label(num_top_classes=5, label="Predictions"),
		gr.Number(label="Prediction time (s)")
	],
	examples=examples_list,
	title=title,
	description=description
)

demo.launch()