<h1> Activation Function (AF)</h1>

## Objective
1. To comprehend the conceptual and mathematical underpinnings of the Activation Function.
2. To execute the Activation Function in a programming language (such as Python).
3. The objective is to examine the attributes and consequences of using the Activation     Function inside neural networks.
## 1.0	Theoretical Understanding
In this paragraph, I would like to explain what is activation function used in neural networks. Equations and graphs of various types of activation functions will be shown in this paragraph also.
### 1.1 Activation Function (AF)
Activation function (AF) is a function used in a neural network to compute the weighted sum of inputs and biases, which is in turn used to decide whether a neuron can be activated or not. The purpose of the activation function is to introduce non-linearity into the output of a neuron. The activation functions are also referred to as transfer functions in some literature.
We know, that the neural network has neurons that work in correspondence with weight, bias, and their respective activation function. In a neural network, we would update the weights and biases of the neurons based on the error at the output. This process is known as back-propagation. Activation functions make the back-propagation possible since the gradients are supplied along with the error to update the weights and biases.
![Screenshot 2024-02-15 at 3 10 26 AM](https://github.com/md-abu-shayid/Deep_Learning_Assignment_01/assets/118624581/bba06750-0f8a-47b5-8c81-aaf7b369cb73)
### 1.2 Usefulness of activation function
A neural network without an activation function is essentially just a linear regression model. The activation function does the non-linear transformation to the input making it capable of learning and performing more complex tasks. <br>
<b>Mathematical proof</b><br>
Suppose we have a Neural net like this:-

![image](https://github.com/md-abu-shayid/Deep_Learning_Assignment_01/assets/118624581/91fa5ef2-b0f6-403a-bad5-90c59947bb4e)

Elements of the diagram are as follows:

<b>Hidden layer i.e. layer 1:</b><br>
z(1) = W(1)X + b(1) a(1) <br>
Here, <br>
•	z(1) is the vectorized output of layer 1 <br>
•	W(1) be the vectorized weights assigned to neurons of the hidden layer i.e. w1, w2, w3 and w4 <br>
•	X be the vectorized input features i.e. i1 and i2 <br>
•	b is the vectorized bias assigned to neurons in the hidden layer i.e. b1 and b2 <br>
•	a(1) is the vectorized form of any linear function. <br>
(Note: We are not considering activation function here)

<b>Layer 2 i.e. output layer:- </b><br>
Note: Input for layer 2 is output from layer 1 <br>
z(2) = W(2)a(1) + b(2)  <br>
a(2) = z(2) <br>
Calculation at the Output layer <br>
z(2) = (W(2) * [W(1)X + b(1)]) + b(2) <br>
z(2) = [W(2) * W(1)] * X + [W(2)*b(1) + b(2)] <br>
Let, 
    [W(2) * W(1)] = W <br>
    [W(2)*b(1) + b(2)] = b <br>
Final output : z(2) = W*X + b; which is again a linear function <br>
This observation results again in a linear function even after applying a hidden layer, hence we can conclude that does not matter how many hidden layers we attach to the neural net, all layers will behave the same way because the composition of two linear functions is a linear function itself. A neuron can not learn with just a linear function attached to it. A non-linear activation function will let it learn as per the difference w.r.t error. <b> Hence we need an activation function. </b>
## Conclusion
The AF is a key component for the training and optimization of neural networks, implemented on different layers of DL architectures, and is used across domains including natural language processing, object detection, classification, and segmentation, etc.
Among all the AF we have discussed above ReLU and its variants should be preferred over sigmoid or tanh activation functions. Also, ReLUs are faster to train. If ReLU is causing neurons to be dead, use Leaky ReLUs or its other variants. Sigmoid and tanh suffer from vanishing gradient problems and should not be used in the hidden layers. ReLUs are best for hidden layers. Activation functions that are easily differentiable and easy to train should be used.
