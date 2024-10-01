# NaturalSceneClassification
Solving the classification issue found here: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
## Introduction:
This repo contains the code required to train a ResNet50 pretrained model to identify different image types based on the Kaggle dataset above.
The model was trained to 92.20% accuracy. The model file I trained is also included.

The model has 6 classes:
- Buildings
- Forest
- Mountains
- Glaciers
- Sea
- Streets

## Method
I trained the model in only 3 epochs as the ResNet50 pretrained model already generalized well to the data set.

I added in a scheduler to adjust the learning rate as training progressed which helped the model generalize better. Early stopping is also in the scripts
but was not needed due to the small epoch number required to reach a good end result.

I also performed some visualizations of the data to see what I was working with before training and to test my ideas for dataaugmentation.

### Here are 5 samples of the training data with no data augmentation:
![alt text](/images/Figure_transform.png "Traning Data No Transforms")

### Here are 5 samples of the validation data with no data augmentation:
![alt text](/images/Figure_validation.png "Validation Data No Transforms")

## Results
### Here is the training over 3 epochs:
![alt text](/images/Figure_1.png "Training Results")

### Here is the confusion matrix of the trained model:
![alt text](/images/Figure_confusion_matrix.png "Confusion Matrix")

### Predicted Outputs
![alt text](/images/Figure_pred_1.png "Predicted Image One")

![alt text](/images/Figure_pred_2.png "Predicted Image Two")

![alt text](/images/Figure_pred_3.png "Predicted Image Three")

## How to use?
Use the following:

```git clone https://github.com/Aeryes/NaturalSceneClassification.git```

Run ```pip install -r requirements.txt```

Place your new images in the data/seg_pred/seg_pred folder

Run ```python predict.py```

