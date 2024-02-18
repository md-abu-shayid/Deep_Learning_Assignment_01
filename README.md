<h1> The ReLU Activation Function: Understanding and Application in the Neural Network</h1>

## Objective

1. To comprehend the conceptual and mathematical underpinnings of the Activation Function.
2. To execute the Activation Function in a programming language (such as Python).
3. The objective is to examine the attributes and consequences of using the Activation     Function inside neural networks.
## 1    Introduction
In the domain of deep learning, a neural network absent of an activation function resembles a linear regression model. These functions drive a neural network’s ability to handle intricate tasks by performing crucial non-linear computations. This article aims to delve into the derivatives, implementations, strengths, and limitations of the Rectified Linear Unit (ReLU) activation function (AF) to help you make an informed choice and infuse non-linearity and precision into your neural network models.
### 1.1 Activation Function (AF)
Activation function (AF) is a function used in a neural network to compute the weighted sum of inputs and biases, which is in turn used to decide whether a neuron can be activated or not. The purpose of the activation function is to introduce non-linearity into the output of a neuron. The activation functions are also referred to as transfer functions in some literature.
We know, that the neural network has neurons that work in correspondence with weight, bias, and their respective activation function. In a neural network, we would update the weights and biases of the neurons based on the error at the output. This process is known as back-propagation. Activation functions make the back-propagation possible since the gradients are supplied along with the error to update the weights and biases.

![Screenshot 2024-02-15 at 3 34 31 AM](https://github.com/md-abu-shayid/Deep_Learning_Assignment_01/assets/118624581/012bdaff-588c-4b29-a2a1-b5bd9197566b)

### 1.2  ReLU Activation Function
ReLU is an activation function that was first presented in (Richard HR Hahnloser et al., 2000) and has solid mathematical and biological foundations. It was shown in 2011 to further enhance deep neural network training. ReLU activation function in the hidden layers has the potential to accelerate the learning rate of different deep neural networks, as demonstrated by (Glorot et al., 2011). Deep neural networks now use the rectified linear unit as their typical activation function. The ReLU function is used to activate all hidden layers; additionally, it prevents saturation and increases the sensitivity of the activation sum. Though it is a nonlinear function that allows the network to find complex nonlinear correlations, it appears and behaves like a linear function. The Sigmoid function is favoured above the Step and Tanh activation functions because of its many advantages. As an illustration, the Sigmoid function limits the values to the interval, has an "S-shaped" characteristic curve, facilitates smoother training than the Step function, and helps to eliminate gradient bias. Tanh is a zero-centered logistic sigmoid function with rescaled outputs in the range (Reshi et al., 2021).
The Rectified Linear Unit (ReLU) function, denoted as ![11](https://github.com/md-abu-shayid/Deep_Learning_Assignment_01/assets/118624581/cafef106-d092-49c9-aeb4-7bd4e161900f) is defined by the following equation:

![Screenshot 2024-02-18 at 11 49 50 PM](https://github.com/md-abu-shayid/Deep_Learning_Assignment_01/assets/118624581/a9dc0fe8-5f0a-4247-87e5-aca464b7434b)

We won't get minimal values when using the ReLU activation function. Rather, it is either 1 or 0 which results in some gradients returning nothing. A significant amount of sparsity was also added to the neural network when the ReLU function was added (Wen et al., 2016). This indicates that there are many zeros in a neural network's active matrix. It is called a sparse neural network when only a portion of activation, say 50%, is saturated. Constant values often demand less space and computational power, therefore this can increase efficiency in terms of time and space complexity (Reshi et al., 2021). Yoshua Bengio et al. discovered that this ReLU component has the previously described efficiency in terms of time and can improve neural network performance (Arpit & Bengio, 2019). Gaussian noise is included in Noisy Relu, which is an extension of ReLU. It is used to tackle computer vision tasks in the constrained Boltzmann machine (Gulcehre et al., 2016). Even yet, the "S-shaped" soft saturation activation function's disappearance of the gradient is resolved by the ReLU function's sparsity. Nevertheless, ReLU's hard saturation of its negative half axis is set to 0, which could result in "neuronal dead" and have an effect on the distribution of its data that is not zero mean. During training, the model may undergo neuronal "death".

Figure 1 demonstrates that the gradients at positive values become constant and no longer disappear. This indicates that the problem of disappearing gradients can be avoided by employing the ReLU activation function. This is why the ReLU activation function can accelerate the learning rate of deep neural networks (Reshi et al., 2021).

![1](https://github.com/md-abu-shayid/Deep_Learning_Assignment_01/assets/118624581/1d8dfa29-dfbc-4ed6-8210-3849e0a11595)

It is evident from Figure 2 that the gradients at positive values become stable and stop disappearing. This indicates that the use of the ReLU activation function can prevent the vanishing gradient issue. This explains why the ReLU activation function can accelerate deep neural network learning.

![2](https://github.com/md-abu-shayid/Deep_Learning_Assignment_01/assets/118624581/0a47a0aa-0e4b-4904-a984-40cc168c3107)

The neural network gains non-linearity from the ReLU activation function, which enables it to recognize intricate patterns and correlations in the input. It helps mitigate the vanishing gradient problem—which can arise with other activation functions like sigmoid or tanh—and is computationally efficient.

## Importance of ReLU in Neural Networks
### Creates non-linearity: 
Complex patterns cannot be learned by neural networks with linear activations. ReLU introduces non-linearity, enabling networks to pick up a variety of features. Quicker instruction and less complicated computations when compared to other activation functions such as tanh and sigmoid. Deep neural networks can now be trained effectively thanks to this. 
### Sparsity: 
A lot of units produce 0 to promote the interpretability of the model and maybe lessen overfitting.

## Output Range of the ReLU Activation Function:
The output range of the ReLU activation function is [0, + ∞). This is because, for all positive values of x, the function outputs the input value i.e. x, and for all negative values of x, the function outputs zero.

So, for any input x, the output f(x) is constrained to be non-negative, with the minimum output being 0.

So, as x varies, the output f(x) is always non-negative, with a minimum value of 0.

## Derivative of the ReLU Activation Function and Backpropagation:
The derivative of the ReLU activation function is crucial for the backpropagation process in training neural networks. The derivative![Screenshot 2024-02-19 at 12 00 30 AM](https://github.com/md-abu-shayid/Deep_Learning_Assignment_01/assets/118624581/e3707a78-9829-47c4-b3a4-6ead008bb643) is given by:

![Screenshot 2024-02-19 at 12 00 17 AM](https://github.com/md-abu-shayid/Deep_Learning_Assignment_01/assets/118624581/4ce452a4-2a6b-45ae-b5d6-9680def3c511)



### Linear activation function
The linear activation function, also known as "no activation," or "identity function" (multiplied x1.0), is where the activation is proportional to the input. The function doesn't do anything to the weighted sum of the input, it simply spits out the value it was given. Mathematically it can be written as
### f(x)=x
A graph of the linear activation function is below.

![Screenshot 2024-02-15 at 4 07 11 AM](https://github.com/md-abu-shayid/Deep_Learning_Assignment_01/assets/118624581/d613150c-aed7-4939-b9d9-573a4fe1cef2)

### 1.3 Usefulness of activation function
A neural network without an activation function is essentially just a linear regression model. The activation function does the non-linear transformation to the input making it capable of learning and performing more complex tasks. <br>
<b>Mathematical proof</b><br>
Suppose we have a Neural net like this:-

![image](https://github.com/md-abu-shayid/Deep_Learning_Assignment_01/assets/118624581/91fa5ef2-b0f6-403a-bad5-90c59947bb4e)

Elements of the diagram are as follows:

<b>Hidden layer i.e. layer 1:</b><br>
z_1= W_1.X + b1 <br>
Here, <br>
•	z_1 is the vectorized output of layer 1 <br>
•	W_1 be the vectorized weights assigned to neurons of the hidden layer i.e. w1, w2, w3 and w4 <br>
•	X be the vectorized input features i.e. x1 and x2 <br>
•	b is the biases assigned to neurons in the hidden layer i.e. b1 and b2 <br>
(Note: We are not considering activation function here)

<b>Layer 2 i.e. output layer:- </b><br>
Note: Input for layer 2 is output z_1 = a_1 from layer 1 <br>
z_2 = W_2.a_1 + b2  <br>
a_2 = z_2 <br>
Calculation at the Output layer <br>
z_2 = (W_2 * [W_1.X + b1]) + b2 <br>
z(2) = [W_1 * W_2] * X + [W_2*b1 + b2] <br>
Let, 
    [W_1 * W_2] = W <br>
    [W_2*b1 + b2] = b <br>
Final output : z(2) = W*X + b; which is again a linear function <br>
This observation results again in a linear function even after applying a hidden layer, hence we can conclude that does not matter how many hidden layers we attach to the neural net, all layers will behave the same way because the composition of two linear functions is a linear function itself. A neuron can not learn with just a linear function attached to it. A non-linear activation function will let it learn as per the difference w.r.t error. <b> Hence we need an activation function. </b>
## Conclusion
The AF is a key component for the training and optimization of neural networks, implemented on different layers of DL architectures, and is used across domains including natural language processing, object detection, classification, and segmentation, etc.
Among all the AF we have discussed above ReLU and its variants should be preferred over sigmoid or tanh activation functions. Also, ReLUs are faster to train. If ReLU is causing neurons to be dead, use Leaky ReLUs or its other variants. Sigmoid and tanh suffer from vanishing gradient problems and should not be used in the hidden layers. ReLUs are best for hidden layers. Activation functions that are easily differentiable and easy to train should be used.
