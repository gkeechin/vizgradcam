# VizGradCAM
VizGradCam is the fastest way to visualize GradCAM in Keras models. Most tutorials or function features similar method but requires the name of the last convolutional layer, performing the upscaling of the heatmap and superimposing it on the original image. In this repository, we aim to combine all of those task without the need for `last_conv_layer_name` by iterating backwards throught the neural network in search of a `Conv2D` class.

This function is inspired by Keras' GradCAM toturial [here](https://keras.io/examples/vision/grad_cam/).

### Usage
The function takes a keras model with loaded weights, an image that is loaded in array form and a boolean flag that determines if the model plots the superimposed image or simply returns the heatmap.
```python
VizGradCAM(model, image, plot_results=True)
```

__Sample Usage__
```python
# Import Function
from gradcam import VizGradCAM

# Load Your Favourite Image
test_img = img_to_array(load_img("electric_guitar.jpeg" , target_size=(224,224)))

# Use The Function - Boom!
VizGradCAM(VGG16(weights="imagenet"), test_img))
```


### Tested / Supported Models
This function works with Keras CNN models and most Keras Applications / Based Models. This means that it will work even if you built used `replace_top` to perform transfer learning on some of the models listed below since in GradCAM, we are looking to targe tthe gradients flowing into the last layer of the CNN.

| Model Architecture |  Support  |  Dimension  |
|--------------------|:---------:|------------:|
VGG16	| ✓	| (224,224)
VGG19	| ✓	| (224,224)
DenseNet121	| ✓	| (224,224)
DenseNet169	| ✓	| (224,224)
ResNet50	| ✓	| (224,224)
ResNet101	| ✓	| (224,224)
ResNet152	| ✓	| (224,224)
ResNet50V2	| ✓	| (224,224)
ResNet101V2	| ✓	| (224,224)
ResNet152V2	| ✓	| (224,224)
MobileNet	| ✓	| (224,224)
MobileNetV2	| ✓	| (224,224)
Xception	| ✓	| (299,299)
InceptionV3	| ✓	| (299,299)
InceptionResNetV2	| ✓	| (299,299)
EfficientNetB0	| ✓	| (224,224)
EfficientNetB1	| ✓	| (240,240)
EfficientNetB2	| ✓	| (260,260)
EfficientNetB3	| ✓	| (300,300)
EfficientNetB4	| ✓	| (380,380)
EfficientNetB5	| ✓	| (456,456)
EfficientNetB6	| ✓	| (528,528)
EfficientNetB7	| ✓	| (600,600)
