# CT-Scan Detection
This project aims to compare the effictiveness of Residual Networks applied to slices of CT-scans
and context-aware NN's for differentiating between Control, Stone, Tumor, and Cyst CT-scans of kidneys.

## Problem Statement
Compare various NN architectures in effectiveness at predicting kidney anomolies as view in CT-scans.

### Why Choose ResNet Architecture?
I chose the ResNet architecture as my baseline for evaluating performance for three main reasons:

1. Strength with Depth

One of many challenges when deciding upon the optimal NN for a clasification task is depth. Adding layers has the advantage of identifying more complex features as lower level identifications from previous layers are compounded upon. The drawbacks can manifest as vanishing/exploding gradients, saturated accuracy, and overfitting causing deeper models to underperform[^ResNet]. A residual NN's resistence to degredation is 

With residual layers a model should not 'forget' what it has learned in previous layers, allowing deeper models to outperform those more shallow up to a point[^ResNet]. I will attempt to identify a layer depth in this project which 

2. Proven Feature Detector

The ResNet architecture has been demonstrated to be general enopugh to classify 1000 classes mainitng the accuracy to win mutliple image classification copmetitions such as ImageNet[^ResNet]. I consider implementing an application specific residual architecture based of ResNet to this task to be a useful experience with an influential CNN.

3. Familiarity

I have [previously](https://github.com/Kyrylo-Bakumenko/Emotion-Recognition#emotion-recognition) applied pre-trained ResNet152 models for emotion classification of facial expressions. To expand my understanding of this architecture I wrote an application specific implementation from scratch for this project. I also intend to 

## My Approach

### Dataset
The I sourced my data from Kaggle[^Dataset]. The dataset was collected from PACS (Picture archiving and communication system) from different hospitals in Dhaka, Bangladesh. It contains the Coronal and Axial images of abdomen and urogram.

For the purposes of the ResNet152 approach I filtered the data to only keep the abdomen images, keeping both those with and without contrast.

![Sample_Kidney][Sample_Kidney]

The frequency of different classes in our data is not uniform and this may lead the model to become biased towords one of the diagnoses, negatively affecting the models training rate and performance.

![Class Distribution][class_distr]

 I take an approack of under-sampling the majority class and over-sampling the minority classes to balance this distribution [^SMOTE]. Specifically I augmented the data by adding in rotated, mirrored, and scaled images from a random crop of the original. In addition to allowing me to balance the ditribution I am able to effectively double the data I have available for training the model.

![Augmented Class Distribution][aug_class_distr]

### The Model (ResNet152)

I am using the ResNet architecture with a layer depth of 152 for my first becnhmark model. In my own implementation I have altered the model to output a last layer of size four to better fit this use case.

To decide which Learning Rate to use for training the model through a Learning Rate Range Test[^LRRT]. From the resulting graph I found that an appropriate learning rate is in the range {10e-5, 10e-3}. I intend to expirement with this range for a cyclical learning rate policy approach in the future, but for this model I kept the learning rate constant: 10e-4.

![Learning Rate Range Test][LRRT]

The model was trained on a single GPU and I was unable to train with a batch size larger than of 16, significantly less than the original ResNet for ImageNet. Despite this is unlikely to be the leading cause for error in the model since a smaller bacths ize is approportiate considering that the entire trainign dataset is smaller as well.

![Training Loss Data][loss_graph]

After training for 32 epochs the accuracy, in-sample, and out-of-sample loss began to plateau so the training was interrupted. Training, howveer, was slowing as soon as the 12th epoch so out of concern for over-fitting the checkpoint of the 12th epoch model was for evaluations. This model achieved an out of sample accuracy of 91.1%

![Training Accuracy Data][accuracy_graph]

### Evaluations

Appart from accuracy, the ResNet model was evaluated with a confusion matrix, ROC-AUC convex curve, and model saliency.

1. Confusion Matrix

The confusion matrix revealed that the model was relatively unbiased between classes with one exception: it had a tendency to misdiagnose stones as cysts.

![Confusion Matrix][confusion_matrix]

2. ROC-AUC Convex-like Curve

As one would expect after viewing the confusion matrix, all of the curves give relatively large AUC's of 0.97, 0.96, 0.95, and 0.89. The curve that has suffered the most being that of the kidney stone class since the model tends to confuse it with the cyst class.

![ROC-Convex Normal][cv_normal]
![ROC-Convex Tumor][cv_tumor]
![ROC-Convex Cyst][cv_cyst]
![ROC-Convex Stone][cv_stone]

3. Saliency Maps

These saliency maps were generated by passing a sliding window across an image. The sliding window patch was either blacked out or whited out, whichever result giving the largest squared difference from the correct classification was taken as the error giving each region of the image a sensitivity value based off of the magnitude of this vector. After normlaization, smoothing with a gaussian filter, and application of a color map, the sensitivty of the model to regions of the image can be returned as a saliency map.

![Saliency Map 1][saliency_1]
![Saliency Map 2][saliency_2]

In these saliency maps it can be observed that the model is generally focusing on the right parts of the image, with two bright regions appearing where the kidneys appear in the dataset. However, the model erroneusly continues to focus on these regions even when the kidneys are obscured as in the second example.

I will implement more saliency maps later in this project, such as class specific saliency maps to visualize how a model's attention changes depending on class. Another approach is to see which pixel values maximize the output of the model per class. As a result one would be able to visualize what the model conceptializes as an ideal member of that class whcih would be helpful to acertain if it is learning the correct features for diangoses [^Saliency].

### In Progress
A similar application for reading CT scans has been implemented in TBI diagnoses. Specificaly I hope to implement a sliding window approach [^SlidingWindow] so as to be able to both maintain a reasonabel input size for the model and not lose the resoltuion of the original image.


[^ResNet]: https://arxiv.org/pdf/1512.03385.pdf
[^Saliency]: https://arxiv.org/pdf/1312.6034.pdf
[^SlidingWindow]: https://arxiv.org/pdf/1806.03265.pdf
[^Dataset]: https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone
[^SMOTE]: https://arxiv.org/abs/1106.1813
[^LRRT]: https://arxiv.org/pdf/1506.01186.pdf

[Sample_Kidney]: imgs/healthy_kidneys.jpg
[class_distr]: imgs/data_count.png
[aug_class_distr]: imgs/augmented_data_count.png
[LRRT]: imgs/LRRT.png
[loss_graph]: imgs/training_loss_augmented_data_model.png
[accuracy_graph]: imgs/training_accuracy_augmented_data_model.png
[confusion_matrix]: imgs/Confusion_Matrix.png
[cv_normal]: imgs/CV_Normal.png
[cv_tumor]: imgs/CV_Tumor.png
[cv_cyst]: imgs/CV_Cyst.png
[cv_stone]: imgs/CV_Stone.png
[saliency_1]: imgs/model_vision_1.png
[saliency_2]: imgs/model_vision_3.png
