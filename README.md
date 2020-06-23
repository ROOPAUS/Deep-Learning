# Deep-Learning
## Basic idea of neural network:

#### Contains 3 layers : 
Input layer, Hidden layer/layers, Output Layer

- Input Layer : Picks up input signals and passes it to next layer

- Hidden Layer/layers : Does calculation and feature extraction

- Output Layer : Delivers the final result

- The interconnections/channels are assigned weights at random

- These weights are multiplied with the input signal ( nodes or activations). Finally bias is added to this weighted sum 

- This weighted sum is calculated and fed to activation function in each layer to decide which nodes to fire

#### Different types of activation functions:

- Sigmoid function : Used when model is predicting probability.

- Threshold function : Used when output depends on a threshold value.

- ReLU ( Rectified Linear Unit) : Gives an output if X is positive, else 0

- Hyperbolic Tangent Function : Similar to sigmoid function, but has range from -1 to 1

#### In case of error:

- Error in the output is back propagated through the network and weights are adjusted to minimize the error rate. This is calculated by a cost function

- Output is compared with the original result and multiple iterations are done to get the maximum accuracy.

- With every iteration, weights at each interconnection are adjusted based on the error

### Actual working of a deep learning model

#### What is a tensor?

- Simply put, tensor is a block of numbers. It can be in 0 dimensional (i.e a single number), 1D (a list), 2D(matrix) or N-dimensional block.

### Types of neural networks:

- Feed Forward Neural Network : Data travels only in 1 direction. Used in speech and vision applications.

- Radial Basis Function Neural Network : Classifies data point based on center point. Used in power restoration systems.

- Kohonen Self-organizing Neural Network : Vectors of random input are input to discrete map consisting of neurons . Used in medical analysis for pattern recognition.

- Recurrent Neural Network : Hidden layer saves its output to be used in future predictions. Used in text to speech conversions.

- Convolutional Neural Network : Input features are taken in batches - like passing through a filter - to remember image in parts. Used in facial recognition.

- Modular NN : Combination of different NN. Still in research phase

### Why deep learning is so transformative?

- No need of feature engineering or extraction

- High performance compared to other models for large datasets

- Works for both structured and unstructured data

### Libraries used for deep learning projects:

- TensorFlow( by Google Brain)

- Keras ( built on top of tensorflow)

- Theano ( an academic project - tightly integrated with Numpy)

- CNTK (by microsoft)

- Caffe (by Facebook)

- PyTorch (by Facebook)

- MxNet ( supported by major cloud providers such as AWS and Azure)

- CuDNN (CUDA DNN)

- deeplearning4j ( distributed DNN for java virtual machines)



Credits -These notes are referred from simplilearn content and from various other blogs



