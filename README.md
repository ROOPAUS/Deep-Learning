# Deep-Learning
## Basic idea of neural network:

#### Perceptron model

- Actual neuron structure is very complex. In the simplest form, it can be shown as consisting of dendrites, nucleus inside cell body and synaptic terminals

- The same can be modelled using numbers. Each input will have a value and a weight associated with. Product(value and weights) of all inputs are summed together and an additional bias is also added, to get an output (if >0, then 1, else -1). This is the perceptron model.

#### Structure of a neural network:

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

- In simple terms, backpropagation is informing the weights about gradient. And the actual changing of weights to minimize the loss is done in the gradient descent i.e decreasing the gradient of the loss.

- Gradient(not the error) in the output is back propagated through the network and weights are adjusted to minimize the error rate. This is calculated by a cost function

- Output is compared with the original result and multiple iterations are done to get the maximum accuracy.

- With every iteration, weights at each interconnection are adjusted based on the error

- To summarize, we start at inputs, generate output. We will have errors. We are going to use back propagation to push the error gradient all the way back telling all the weights how it should move and we will adjust the weights up or down.

#### Learning rate:

- The size of step that we take to adjust the weights is known as the learning rate ( denoted by eta)

- For a error curve with a single minimum point, learning rate can be used without any hassle.

- But imagine, for a curve with ups and downs, the learning rate makes the gradient go downhill to the first minimum and settles. If we give a large learning rate, it can overshoot to the next uphill directly without moving down the hill (i.e we may end overshooting the global minimum). This is not the proper way to do it.

- So, one thing we can do is the "decay schedule"

- #### Decay schedule involves decreasing the learning rate after each epoch (Epoch refers to one pass of the training loop)

- Another way to do it is to use momentum or inertia instead of normal gradient descent

#### Momentum:

- The velocity to reach a point A (lets say v) is scaled up by a temporal or time element denoted by gamma. (Gamma value range from 0 to 1, 0.9 being the one usually used). Now we reach point B. This v* gamma is the momentum.

- We find the gradient of point B and scale it using learning rate . And then we add the previous momentum (i.e gamma * v)

- This is almost like adding the previous gradient to the present gradient (not exactly the same, but easier to grasp the concept)

- The momentum and learning rates are "set" by trial and error. It depends on our data and our network.

#### Nestorov Momentum (Nesterov Accelerated Gradient NAG):

- This is similar to normal momentum update. The only difference is in the gradient used in both of these.

- In normal momentum, at every time step the velocity is updated according to the local gradient and is then applied to the parameters.

- That is, [v <sub>(t+1)</sub> = ($\gamma$ * v) - ($\eta$ * local gradient)]. Then we calculate C as [($\theta$ + v<sub>(t+1)</sub>)] . Here local gradient refers to gradient at B and $\theta$ is the parameters of B referring to weights and bias at B

- But in Nestorov Momentum, we treat the future approximate position ["Point B" + ($\gamma$ * v)] as a “lookahead” - this is a point in the vicinity of where we are soon going to end up. Note, its Point B and not gradient at B.

- [Lookahead point = "Point B" + ($\gamma$ * v)]

- [v <sub>(t+1)</sub> = ($\gamma$ * v) - ($\eta$ * gradient at lookahead point )]

- Then we calculate C as [($\theta$ + v<sub>(t+1)</sub>)]

- This is to reduce the influence of momentum in determining the direction of move.

#### Hyperparameters:

- It is the weights and bias that change, so they are parameters. The learning rate and momentum are set these and then train, again set and then train in a repeated fashion and hence are called hyperparameters.

#### Convergence - an open ended question!

- A model converges when its loss actually moves towards a minima (local or global) with a decreasing trend ( meaning it properly responds to inputs)

- The hyperparameters all help in minimizing the loss and helps the system correctly predict the output. But is the system strictly converging?

- The answer is NO. Only by tweaking the hyperparameters again and again, we reach a minimal loss. Therefore, except for the single perceptron model with a single neuron, convergence is not seen in any of these networks.

- It is using tweaking and our own sense of judgement that we finally reach a minimal loss stage.

#### Overfitting:

- We split the input data into 3 sets : training set, validation set amd test set

- In each epoch we train with the training data, and then run through the validation data ( which is the proxy for  test set)

- To improve accuracy, we again train the training data and run through the validation data

- This happens until we reach the accuracy we want. Then we bring out the test set and run through it. If it gives the same accuracy as validation set, then its fine, else we start all over again.

- After running the training and validation sets many times, slowly the error curve for training set starts flattening out, but the error curve for validation set shoots up. This is because of overfitting. The model starts to learn tricks to reach the output directly sometimes giving wrong answer for another set of inputs. ( Eg: Puddle tail)

#### Regularization:

- To reduce overfitting, we use certain algorithms called Regularization algorithms

- One of the most famous regularisation algorithm is dropout

#### Dropout:

- Suppose we have a set of neurons, and one of those is looking for a particular trait in the image, and when it sees that trait, it dominates and propagates through the network to give an output which sometimes can be wrong.

- So to avoid this, at the beginning of each epoch, we randomly select a set of neurons and disconnect it from the network, so that they dont participate in the forward and backward progation and their weights also wont get updated. Then at the next epoch, we connect those neurons back and pick another set of neurons and disconnect them.This is done with different set of neurons each time.

- This is a easy, simple to implement and fast out of all regularization algorithms.

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


### Actual implementation of a deep learning model

#### What is a tensor?

- Simply put, tensor is a block of numbers. It can be in 0 dimensional (i.e a single number), 1D (a list), 2D(matrix) or N-dimensional block.


Credits -These notes are referred from simplilearn content and from various other blogs
https://dominikschmidt.xyz/nesterov-momentum/
https://www.youtube.com/watch?v=kPrHqQzCkg0
https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3
https://www.youtube.com/watch?v=odlgtjXduVg
https://mlfromscratch.com/optimizers-explained/#/
https://cs231n.github.io/neural-networks-3/#sgd
https://www.youtube.com/watch?v=r0Ogt-q956I

HOG and PCA
