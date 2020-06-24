# Deep-Learning
## Basic idea of neural network:

#### Contains 3 layers : 
Input layer, Hidden layer/layers, Output Layer

- Input Layer : Picks up input signals and passes it to next layer. It is a buffer where input value sits. No processing is done here.

- Hidden Layer/layers : Does calculation and feature extraction

- Output Layer : Delivers the final result. Unlike input layer, the output layer has computation.

- The interconnections/channels are assigned weights at random

- These weights are multiplied with the input signal ( nodes or activations). Finally bias is added to this weighted sum 

- This weighted sum is calculated and fed to activation function in each layer to decide which nodes to fire. 

#### Why do we need this activation function?

- Each hidden layer is actually the weighted sum of the previous layer. Without activation function present, the output that we get after all the processing time is all just addition. That means, we can remove all the hidden layers, it will be just input layer and output layer. But this isnt how the actual neuron works. Based on the threshold value the outgoing signal is determined and sent. Similarly, we can include activation functions in our model to determine what the output to the next layer should be, thus preventing possible collapse.

#### Different types of activation functions:

- Sigmoid function : Used when model is predicting probability.

- Threshold function : Used when output depends on a threshold value.

- ReLU ( Rectified Linear Unit) : Gives an output if X is positive, else 0

- Hyperbolic Tangent Function : Similar to sigmoid function, but has range from -1 to 1

#### Is softmax an activation function?

- No, softmax is not exactly an activation function, but it almost serves the same purpose.

- Softmax layer comes at the end of the classification network. The output values from the output neurons are output scores which are not probabilities. These scores are infact negative logarithm of likelihoods. Inorder to convert these scores to probabilities, the softmax layer is used ( i.e squishing the dynamic range from 0 to 1).


#### Gradient Descent:

- The predicted value is compared with the actual label using error formula or loss formula

- #### Loss function : It is the measurement of error which defines the precision lost when comparing predicted output to the actual output. Simply put, loss function is the loss of accuracy.

- #### loss = [(actual output)-(predicted output)]^2

- Here, inorder to get the most accurate output, the loss should be minimum ( 0 in ideal case). So how do we find an input that minimizes the value of a function? In calculus, we use the derivates to find the minima of a function explicitly( where slope= 0 and second derivative>0). But in complex functions in deep learning, involving large number of inputs, this method is not feasible.

- In this case, we plot a graph of weight versus loss. A random point on this curve is chosen and the slope at this point is calculated.

- Shift the input to the left if the slope is positive, i.e a positive slope indicates an increase in weight.

- Shift the input to the right if the slope is negative, i.e a negative slope indicates a decrease in weight.

- A zero slope indicates the appropriate weight. Our aim is to reach a point where the slope is zero.

- This method of finding minimum of a function is known as the gradient descent.

#### Back Propagation:

- Error in the output is back propagated through the network and weights are adjusted to minimize the error rate. This is calculated by a cost function

- Output is compared with the original result and multiple iterations are done to get the maximum accuracy.

- With every iteration, weights at each interconnection are adjusted based on the error

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


### Actual working of a deep learning model

#### What is a tensor?

- Simply put, tensor is a block of numbers. It can be in 0 dimensional (i.e a single number), 1D (a list), 2D(matrix) or N-dimensional block.

#### Perceptron model

- Actual neuron structure is very complex. In the simplest form, it can be shown as consisting of dendrites, nucleus inside cell body and synaptic terminals

- The same can be modelled using numbers. Each input will have a value and a weight associated with. Product(value and weights) of all inputs are summed together and an additional bias is also added, to get an output (if >0, then 1, else -1). This is the perceptron model.

Credits -These notes are referred from simplilearn content and from various other blogs



