# fake-news-cnn-lstm-project
This research aims to develop a model for detecting fake news by analyzing the stance of articles and the reliability of their sources. Stance features, defined as unsupported claims, are employed to assess the authenticity of news. The model classifies news articles into categories such as agree, disagree, unrelated, or discuss based on the alignment between headlines and their assigned body content. The proposed methodology involves using keywords to identify relevant articles and leveraging PCA and Chi-square for feature reduction. PCA, a popular statistical technique, is employed to enhance classifier performance by transforming the original variables into a subset with reduced dimensions.
## Table of Contents
* Prerequisites
  * Environment
  * Technologies used
  * Dataset
* System Architecture

## Prerequisites
### Environment
* Python 3 Environment 
* Python modules required:Tensorflow, NumPy,Pandas,Flask,Django,Matplotlib, Scikit-learn, Keras
* Web Browser
### Technologies used
* Anaconda Jupyter Notebook
### Dataset
This research utilizes datasets from diverse sources such as Kaggle, Reuters, and fact-checking websites like News Trends. Specifically, the study works with the Reuters dataset, a widely used benchmark for document classification, comprising short newswires and topics published in 1986. This dataset includes 46 different topics, with each topic having at least 10 examples in the training set, making it a simple yet effective dataset for text classification.
![image](https://github.com/poornikabonam/fake-news-cnn-lstm-project/assets/97566249/2e9c570c-396f-44c4-b242-a642c858a9c6)


## System Architecture
The system architecture for text classification involves the integration of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to achieve accurate and efficient classification of text data. The architecture follows a multi-layered approach, encompassing both convolutional and recurrent layers for comprehensive feature extraction and context understanding.
### Data Input:
        Text data from various sources, such as news articles or social media content, serves as the input to the system.
        The raw text is preprocessed to convert it into a format suitable for analysis, including tokenization, stemming, and removal of stop words.

### Embedding Layer:
        The initial layer involves embedding words into low-dimensional vectors, enabling the neural network to work with continuous vector representations of words.
        This layer captures semantic relationships between words and prepares the input for further processing.

### Convolutional Neural Network (CNN) Layer:
        A one-dimensional CNN (Conv1D) is employed to identify patterns and features within the text.
        Multiple filters of varying sizes slide over the embedded word vectors, detecting patterns like word n-grams.
        Convolution operations result in feature maps representing relevant features in the input data.

### Pooling Layer:
        Max-pooling is applied to the output of the convolutional layer, reducing the dimensionality of the feature maps.
        Pooling helps retain essential information while discarding less relevant details, facilitating effective feature extraction.

### Long Short-Term Memory (LSTM) Layer:
        The LSTM network is introduced to capture contextual information and address the vanishing gradient problem associated with traditional recurrent networks.
        LSTM cells maintain memory blocks that retain information over prolonged sequences, allowing the model to learn long-term dependencies.

### Output Layer:
        The final layer involves classifying the processed features using a softmax layer.
        The softmax layer assigns probabilities to different classes, enabling the system to predict the category or sentiment of the input text.
### Training and Optimization:
        The entire architecture is trained using machine learning algorithms, with the model adjusting its parameters to minimize classification errors.
        Optimization techniques, such as model checkpointing and dropout regularization, are applied to enhance the model's generalization capabilities.

### System Design and User Interaction:
        The architecture is designed with a user-friendly interface, incorporating logical and physical design considerations.
        Input and output screens are crafted to be precise and compact, ensuring a seamless user experience.
