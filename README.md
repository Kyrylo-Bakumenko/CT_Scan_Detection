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






I plan on improving saliency based off of [^Saliency]
sliding window approach [^SlidingWindow]


[^ResNet]: https://arxiv.org/pdf/1512.03385.pdf
[^Saliency]: https://arxiv.org/pdf/1312.6034.pdf
[^SlidingWindow]: https://arxiv.org/pdf/1806.03265.pdf
[^Dataset]: https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone
