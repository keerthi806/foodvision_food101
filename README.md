# FoodVision

FoodVision is an image-classification project built with PyTorch and
torchvision. The project explores transfer learning with pretrained
computer-vision models and turns the trained model into a Gradio web
application.

The notebook covers two stages:

-   **FoodVision Mini** --- a lightweight 3-class classifier for
    **pizza, steak, and sushi**, with a focus on accuracy and low
    inference latency.
-   **FoodVision Big** --- an expanded classifier trained on the
    **Food101** dataset with **101 food classes**, using data
    augmentation to help reduce overfitting.

## Hugging Face Space

**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/kb0968237/foodvision_Food101)

> Replace `<YOUR_HF_SPACE_URL>` with the URL of your deployed Hugging
> Face Space.

------------------------------------------------------------------------

## Features

-   Transfer learning using pretrained **EfficientNet-B2**
-   Comparison with **Vision Transformer (ViT-B/16)**
-   Image preprocessing using the pretrained model's recommended
    transforms
-   Evaluation of model accuracy, size, and inference speed
-   Gradio interface for interactive image classification
-   Example images included in the deployment package
-   Hugging Face Spaces deployment setup

## Project Overview

The initial deployment goal for FoodVision Mini was to build a model
that performs **well and fast**:

-   Target accuracy: approximately **95%**
-   Inference as close to real time as possible
-   Low latency for individual image predictions

Two pretrained architectures were experimented with:

1.  **EfficientNet-B2 feature extractor**
2.  **ViT-B/16 feature extractor**

For both approaches, the pretrained backbone is frozen and a
task-specific classification head is added.

------------------------------------------------------------------------

## FoodVision Mini

FoodVision Mini classifies an input image into one of three categories:

  Class   Label
  ------- ---------
  🍕      `pizza`
  🥩      `steak`
  🍣      `sushi`

### Model

The project uses pretrained EfficientNet-B2 weights from torchvision.
The original classifier is replaced with:

``` python
nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1408, out_features=3)
)
```

The pretrained feature extractor is frozen, so training focuses on the
new classification layer.

### Inference

The Gradio demo accepts a PIL image and returns:

-   Prediction probabilities for the three classes
-   Prediction time in seconds

------------------------------------------------------------------------

## FoodVision Big

FoodVision Big extends the project to the **Food101** dataset and
predicts one of **101 food categories**.

The notebook uses:

-   `torchvision.datasets.Food101`
-   A 20% subset of the available train/test data for experimentation
-   Batch size of `32`
-   EfficientNet-B2 as the feature extractor
-   `TrivialAugmentWide` for training-time augmentation
-   Cross-entropy loss with label smoothing
-   Adam optimizer
-   5 training epochs

### Why data augmentation?

The notebook introduces data augmentation because larger datasets and
larger models can make **overfitting** more of a concern.
`TrivialAugmentWide` is applied to the training images while the
pretrained EfficientNet-B2 transforms are used for evaluation.

### Label smoothing

FoodVision Big uses:

``` python
nn.CrossEntropyLoss(label_smoothing=0.1)
```

Label smoothing acts as a regularization technique and can help prevent
the model from becoming excessively confident in a single class.

------------------------------------------------------------------------

## Architecture

The deployment model follows this general pipeline:

``` text
Input Image
    │
    ▼
EfficientNet-B2 preprocessing
    │
    ▼
Pretrained EfficientNet-B2 backbone
    │
    ▼
Custom classification head
    │
    ▼
Softmax probabilities
    │
    ▼
Top predictions + inference time
```

The FoodVision Big deployment loads the trained model weights from a
`.pth` file and reads the 101 class names from `class_names.txt`.

------------------------------------------------------------------------

## 📁 Deployment Structure

### FoodVision Mini

``` text
foodvision_mini/
├── 09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth
├── app.py
├── examples/
│   ├── example_1.jpg
│   ├── example_2.jpg
│   └── example_3.jpg
├── model.py
└── requirements.txt
```

### FoodVision Big

``` text
foodvision_big/
├── 09_effnetB2_feature_extractor_food101_20_percent.pth
├── app.py
├── class_names.txt
├── examples/
│   ├── chocolate_mousse.jpeg
│   ├── grilled_salmon.jpeg
│   └── miso_soup.jpeg
├── model.py
└── requirements.txt
```

------------------------------------------------------------------------

## Installation

Create a Python environment and install the project dependencies:

``` bash
pip install torch torchvision gradio
```

For the FoodVision Big Space, the deployment requirements also include:

``` bash
pip install spaces
```

The notebook's deployment `requirements.txt` contains:

``` text
torch
torchvision
gradio
spaces
```

------------------------------------------------------------------------

## Run the Gradio App Locally

From the deployment directory:

``` bash
python app.py
```

The application launches a Gradio interface where you can upload a food
image and receive the model's predictions.

------------------------------------------------------------------------

## Model Evaluation

The notebook compares EfficientNet-B2 and ViT-B/16 using:

-   Test loss
-   Test accuracy
-   Number of parameters
-   Model size
-   Average inference time per prediction

It also visualizes the **performance vs. inference-speed trade-off** to
help determine which architecture is better suited for a fast deployment
scenario.

> Exact benchmark values are intentionally not hard-coded here because
> they depend on the executed notebook results and runtime environment.

------------------------------------------------------------------------

## Training Details

### EfficientNet-B2

The FoodVision Big experiment uses:

``` python
NUM_EPOCHS = 5

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.Adam(
    lr=1e-3,
    params=effnetB2_food101.parameters()
)
```

The model is trained using the notebook's modular training engine.

### ViT-B/16

The FoodVision Mini experiments also include ViT-B/16 as a comparison
model. Its pretrained parameters are frozen and a new classification
head is used for the three target classes.

------------------------------------------------------------------------

## Model Files

The trained models are saved as PyTorch state dictionaries (`.pth`):

-   `09_pretrained_effnetB2_feature_extractor_pizza_steak_sushi_20_percent.pth`
-   `09_pretrained_ViTB16_feature_extractor_pizza_steak_sushi_20_percent.pth`
-   `09_effnetB2_feature_extractor_food101_20_percent.pth`

The FoodVision Big deployment uses the EfficientNet-B2 Food101 model.

------------------------------------------------------------------------

## Example Predictions

The Gradio applications include example images so the deployed interface
can be tested without uploading an image manually.

FoodVision Big includes example images such as:

-   Chocolate mousse
-   Grilled salmon
-   Miso soup

These examples demonstrate the type of food images the model can
process.

------------------------------------------------------------------------

## Tech Stack

-   **Python**
-   **PyTorch**
-   **Torchvision**
-   **Gradio**
-   **Food101 dataset**
-   **EfficientNet-B2**
-   **Vision Transformer (ViT-B/16)**
-   **Hugging Face Spaces**

------------------------------------------------------------------------

## Project Workflow

``` text
Dataset
   │
   ▼
Data preprocessing
   │
   ▼
Pretrained model
   │
   ▼
Replace classification head
   │
   ▼
Freeze backbone
   │
   ▼
Train classifier
   │
   ▼
Evaluate accuracy + speed
   │
   ▼
Save model weights
   │
   ▼
Build Gradio application
   │
   ▼
Deploy to Hugging Face Spaces
```

------------------------------------------------------------------------

## Future Improvements

Potential next steps include:

-   Train on the full Food101 dataset
-   Experiment with more extensive augmentation strategies
-   Tune learning rate and other hyperparameters
-   Improve inference speed
-   Compare additional pretrained architectures
-   Add confidence thresholds and better error handling
-   Improve the Gradio UI
-   Add more deployment-focused documentation

------------------------------------------------------------------------

## Acknowledgements

This project uses pretrained models and datasets provided through the
PyTorch/torchvision ecosystem, including EfficientNet-B2, ViT-B/16, and
Food101.

------------------------------------------------------------------------
