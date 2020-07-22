---
title: 'Deep learning theory at ICML 2020'
date: 2020-07-22
permalink: /posts/2020/07/22/dl_theory_icml_2020/

---
{% include base_path %}

This post is intended to serve as a quick summary of the advancements made in theoretical deep learning that were presented at ICML 2020. You are encouraged to check the paper (linked with the title) that interests you the most. In this blog, I discuss four of my favorites from the conference:

1. [Rethinking bias-variance trade-off for generalization of neural networks](#rethinking-bias-variance-trade-off-for-generalization-of-neural-networks)
2. [Dynamics of Deep Neural Networks and Neural Tangent Hierarchy](#dynamics-of-deep-neural-networks-and-neural-tangent-hierarchy)
3. [Proving the lottery ticket hypothesis: Pruning is all you need](#proving-the-lottery-ticket-hypothesis-pruning-is-all-you-need)
4. [Linear Mode Connectivity and the Lottery Ticket Hypothesis](#linear-mode-connectivity-and-the-lottery-ticket-hypothesis)
 

## [Rethinking bias-variance trade-off for generalization of neural networks](https://proceedings.icml.cc/static/paper_files/icml/2020/2946-Paper.pdf)

Neural networks are heavily overparameterized models and if we are to go by classical theory, such models shouldn't generalize. This paper is a step forward in the goal of demystifying the generalization puzzle in deep learning. Concretely, this paper studies the bias-variance curves for neural networks with varying number of parameters and shows that variance behaves differently to what the classical theory suggests.

### Classical idea: _Monotonic bias, Monotonic variance_

We know that $\textrm{Risk}$ can be decomposed into $\textrm{Bias}^2$  and $\textrm{Variance}$. Classical theory tells us that the $\textrm{Bias}^2$ decreases montonically with model complexity, while $\textrm{Variance}$ increases montonically. $\textrm{Risk}$ initially decreases as the model's capacity increases, but then starts increasing in the overfitting regime. Here, the increase in $\textrm{Variance}$ dominates the decrease in $\textrm{Bias}^2$ leading to a U-shaped $\textrm{Risk}$ curve.

{% include image.html url="https://i.ibb.co/92R65S7/1.png" description="Classical idea of bias-variance trade-off" width="50%" %}  

But this U-shaped risk curve isn't seen in deep learning. Increasing width or depth of neural networks usually results in decreased risk. One of the explanation for such a behavior is [_Double descent_](https://arxiv.org/abs/1812.11118), which extends the classical theory for overparameterized models like neural networks. Double descent introduces a new regime beyond the overfitting regime, called as the interpolating regime, where risk decreases with model complexity. Thus, double descent suggests a peak in risk at the transition between the overfitting and the interpolating regime.

{% include image.html url="https://i.ibb.co/D70SZhX/2.png" description="Extension of classical theory for deep learning" width="100%" %}  


We typically observe montonically decreasing risk (no peaks) or a small bump (short overfitting regime) with real-world data. However, when label noise is injected in data double descent curves are clearly seen. 

### Proposed idea: _Unimodal variance_ (_along with Montonic bias_)

Now, the authors are interested in the behavior of bias and variance curves in deep learning. They propose that the variance curve is actually unimodal and not monotonic, while the bias curve is montonically decreasing. The unimodal behavior of variance essentially means that variance increases, peaks and then decreases with model complexity. Depending on the relative magnitude of the $\textrm{Bias}^2$ and $\textrm{Variance}$, one the following $\textrm{Risk}$ curves can be observed:

{% include image.html url="https://i.ibb.co/fN2SjD4/3.png" description="Risk error curve can take three different forms depending on the relative magnitude of bias and variance" width="100%" %}  

The unimodal behavior of variance explains why we see double descent. The risk curve depicted in Case 2 in the figure above exhibits double descent behavior. 

By computing the value of variance for models of varying capacity across multiple computer vision datasets, the authors empirically prove  that variance is indeed unimodal. However, the authors don't provide an explanation as to why variance behaves in this manner. 

{% include image.html url="https://i.ibb.co/PmPL0Vv/4.png" description="Experimental Results validating the occurence of unimodal variance." width="100%" %}  

### Computing the bias and variance: _Devil is in the details_ 

We know that, $\textrm{Risk} = \textrm{Bias}^2 + \textrm{Variance}$. You can take a look at the paper for the exact bias-variance decomposition for MSE and Cross Entropy losses. 

*One of the ways* $\textrm{Variance}$ is estimated is described below:
- Split the training dataset, $D$ into two halves $D_1$ and $D_2$.
- Train classifiers $f_1$ and $f_2$ on $D_1$ and $D_2$ respectively.
- Unbiased estimator of $\textrm{Variance}$ is given by $\frac{1}{2} (f_1(x) - f_2(x))^2$. 
- Average the above estimate across the entire test set. 
- Repeat the above steps for different splits of $D$ and average across the results.
- $\textrm{Bias}^2$ is obtained by subtracting the $\textrm{Variance}$ from $\textrm{Risk}$

### Unimodal variance as explantion for Double descent behavior

Double descent is clearly seen when label noise is injected to real world data. Otherwise, it can be very small and we could miss it easily. The figure below shows that double descent becomes more and more prominent with increasing label noise.

{% include image.html url="https://i.ibb.co/qnMJsDY/5.png" description="Appearance of double descent behavior with increasing label noise" width="60%" %}  


The authors show that with increasing label noise, the $\textrm{Variance}$ increases and peaks at higher value, increasing the value of risk's peak higher and which causes a more prominent double descent behavior. 

{% include image.html url="https://i.ibb.co/PcYJD8T/6.png" description="Variance increases with label noise and double descent behavior is observed" width="100%" %}  


### Random design v/s Fixed design

*Random design*

All the experiments and the observed behavior is for the _random design_ setting, in which the expectation in $\textrm{Bias}^2$ and $\textrm{Variance}$ is over different training sets, $\mathcal{T}$. This is the usual way of doing things in machine learning.

<center> $ \displaystyle \mathbb{E}_{x, y} \mathbb{E}_{\mathcal{T}}[(y - f(x, \mathcal{T}))^2] = \mathbb{E}_{x, y}[(y - \bar f(x))^2] + \mathbb{E}_{x} \mathbb{E}_{\mathcal{T}}[(f(x, \mathcal{T}) - \bar f(x))^2] $ </center>

where  $\bar f(x) = \mathbb{E}_{\mathcal{T}}[f(x, \mathcal{T})]$.

The first term is $\textrm{Bias}^2$ while the second one is $\textrm{Variance}$.


*Fixed design*


But theoretical analysis is usually done in the _fixed design_ setting. The covariates $x_i$ (training instances) are fixed and the randomness comes from $y_i \sim \mathbb{P}[Y \vert X = x_i]$.  Typically, $y_i = f_0(x) + \epsilon_i  \; \textrm{where} \; \epsilon_i \sim \mathcal{N}(0, \sigma_i^2)$.

<center> $\displaystyle \mathbb{E}_{\epsilon} [(y - f(x, \mathcal{T}_{\epsilon}))^2] = (y - \bar{f}(x))^2 + \mathbb{E}_{\epsilon}[(f(x, \mathcal{T}_{\epsilon}) - \bar{f}(x))^2]$ </center> <br/>

Usually in the _fixed design_ setting, a larger bias and a smaller variance exists (I don't understand why). In this _fixed design_ setting, monotonic bias and unimodal variance don't necessarily hold. Refer to [Mei & Montanari, 2019](https://arxiv.org/abs/1908.05355) for more.

{% include image.html url="https://i.ibb.co/smLGtKv/8.png" description="Bias variance curve in the fixed design setting" width="45%" %}  

### Miscellaneous

Increased bias explains why risk curve moves upwards for out-of-distribution samples. 

{% include image.html url="https://i.ibb.co/KjJkDmD/7.png" description=" " width="100%" %}  


## [Dynamics of Deep Neural Networks and Neural Tangent Hierarchy](https://proceedings.icml.cc/static/paper_files/icml/2020/1356-Paper.pdf)

The motivation behind this paper is to understand the optimization process in neural networks. Neural networks are trained with loss functions that are highly non convex with respect to their parameters. How is that Stochastic Gradient Descent (SGD) is able to efficiently converge to solutions that generalize well? To answer this question, we can study the dynamics of the neural networks, *i.e.* how the neural network changes during the training process, which is what this paper does. 

### Neural Tangent Kernel

Neural networks are complex objects and often assumptions need to be made for mathematical convenience. Here, neural networks are defined in a slightly modified fashion.

**Neural network:**
	$\displaystyle \quad f(x, \theta) = a^T x^{(H)}, \quad  x^{(l)} = \frac{1}{\sqrt m} \sigma(W^{(l)}, x^{(l-1)}) \quad l =1, \cdots, H$  and  $\theta = [\textrm{vec}(W^{(1)}), \cdots, \textrm{vec}(W^{(H)}), a$].

The MSE loss function is considered,
**Loss function:** $\displaystyle L(\theta) = \frac{1}{2n}\sum_{1}^{n} (f(x_i, \theta) - y_i)^2$

The dynamics of the parameters are given by their gradient wrt the loss function, <br/>
<center> $\displaystyle \partial_{t} W^{(l)}_{t} = - \; \partial_{W^{(l)}} L(\theta_t) \quad \partial_{t} a_{t} = - \; \partial_{a} L(\theta_t)$ </center>


Recent works study the training dynamic in the _trajectory_ space instead of _parameter_ space as the trajectory space is compact $(\mathbb R^n)$ and easier to interpret.

**Trajectory space:** $(f(x_1, \theta_t), f(x_2, \theta_t), \cdots, f(x_n, \theta_t))$

**Parameter space:** $\theta_t = (W^{(1)}_t, W^{(2)}_t, \cdots, W^{(H)}_t, a_t)$

Now the dynamic in the trajectory space can we worked out as follows:

{% include image.html url="https://i.ibb.co/9thnMsQ/dynamic.png" description=" " width="70%" %}  

where, 

<center> $\displaystyle K^{(2)}_{t} (x, x') = \langle \nabla_{\theta} f(x, \theta_t), \nabla_{\theta} f(x', \theta_t)  \rangle$ </center> <br/>

is the _Neural Tangent Kernel_ (NTK). 

The following theorems are known for the Neural Tangent Kernel:

**Theorem by [Jacot *et. al*, '18](https://arxiv.org/pdf/1806.07572.pdf):** $K^{(2)}_t(\cdot, \cdot) = K^{(2)}_0(\cdot, \cdot)$ when $m$ approaches infinity. 

**Theorem by Du *et. al*, '18 ([a](https://arxiv.org/pdf/1810.02054.pdf) ; [b](https://arxiv.org/abs/1811.03804)):** $K^{(2)}_t(\cdot, \cdot) \approx K^{(2)}_0(\cdot, \cdot)$ when $m > n^4.$

In the infinite width case, the training dynamic becomes analytically solvable and it can be seen that an infinitely wide neural network behaves like a kernel regression model. 

<!--
Neural Tangent Kernel was introduced by [Jacot et. al](https://arxiv.org/pdf/1806.07572.pdf). Following theorem is from [Du et. al](https://arxiv.org/abs/1811.03804):
>[Informally] Given a $L$-layer Neural network, $f(W, x)$ with Relu activations with $m$ neurons in every layer and $N$ distinct training examples. If the following conditions are satifisfied,
 > - Over-parametrization: $m \ge poly(N, L)$
 > 
 > - Initilization: $W_0 \sim \mathcal{N}(0, \frac{2}{m})$
>  - Small learning rate: $\eta \ll \frac{1}{\sqrt m}$
>  
> Then, SGD finds the solution $W^*$ efficiently with small training loss, $o(1)$ such that
> 
> $f(W^*, x) = f(W, x) + \langle \nabla_Wf(W_o, x), W^*-W\rangle + o(1)$
--->

***HOWEVER,***

NTKs perform worse than real-life deep learning ([Arora et. al '19: Convolutional NTKs](https://arxiv.org/pdf/1904.11955.pdf)). Some improvements have been made ([Enhanced CNTKs](https://arxiv.org/abs/1907.04595)).

The authors say that,
> It is possible to show that the class of finite width neural networks is more expressive than the limiting NTK. It has been constructed in (Ghorbani *et al.*, 2019; Yehudai & Shamir, 2019; **Allen-Zhu & Li, 2019**) that there are simple functions that can be efficiently learnt by finite width neural networks, but not the kernel regression using the limiting NTK.

The observed disparity stems from the fact that NTK varies over time in the finite width case. We can obtain equations for the dynamics of the NTK:

<center> $\displaystyle \partial_t K^{(2)}_t(x_{i_1}, x_{i_2}) = -\frac{1}{n} \sum_{j=1}^n K^{(3)}(x_{i_1}, x_{i_2}, x_j)(f(x_j, \theta_t) - y_j)$ </center>

where, <br/>

<center> $\displaystyle K^{(3)}(x_{i_1}, x_{i_2}, x_j) = \langle \nabla^2_\theta f(x_{i_1}, \theta_t) \; \nabla_\theta f(x_j, \theta_t), \nabla_\theta f(x_{i_2}, \theta_t) \rangle$ </center>


<br/>

If we keep continuing,

<center> $\displaystyle\partial_t K^{(r)}_t(x_{i_1}, x_{i_2}, \cdots, x_{i_r}) = -\frac{1}{n} \sum_{j=1}^n K^{(r+1)}_t(x_{i_1}, x_{i_2}, \cdots, x_{i_r}, x_j)(f(x_j, \theta_t) - y_j) \quad r=2,3,4 \cdots$ </center>

we obtain the _Neural Tangent Hierarchy_, which is given by the equations above.

### Truncated Neural Tangent Hierarchy

Let's truncate this hierarchy to $p$.

For $2 \le r\le p-1:$ <br/>

- <center> $\displaystyle \partial_t \tilde{K}^{(r)}_t (x_{i_1}, x_{i_2}, \cdots, x_{i_r}) = - \frac{1}{n} \sum_{j=1}^{n} \tilde{K}^{(r+1)}_t(x_{i_1}, x_{i_2}, \cdots, x_{i_r}, x_j)(\tilde f(x_j, \theta_t) - y_j)$ </center> <br/>

- <center> $\displaystyle \partial_t \tilde K^{(p)}_t(x, x') = 0$ </center> <br/> 

$\tilde f(x_j, \theta_0) = f(x_j, \theta_0)$ and $\tilde K^{(r)}_0 = K^{(r)}_0$

Note that for $p=2$, we have the NTK theorems by Jacot *et. al.* and Du *et. al.* 

#### Main theorem of the paper

Let $p^* \ge 2$ and $\tilde f$ be the solution to the truncated Neural tangent hierarchy at $p^*$  
 
<center> $\quad \forall p \; \textrm{s.t.} \; 2 \le p \le p^*$ and </center> <br/>
 
<center> $\quad \forall t \; \textrm{s.t} \; t \le \min \Big( \sqrt{\frac{m}{n}}, m^{\frac{p^*}{2 (p^* + 1)}} \Big)$ </center> <br/>

we have,
 
<center> $\displaystyle \Big ( \frac{1}{n} \sum_{j=1}^{n}(f(x_j, \theta_t) - \tilde f(x_j, \theta_t))^2 \Big)^{\frac{1}{2}} \le \Big(\frac{t}{\sqrt m} \Big)^p \; \min(t, n)$ </center> <br/>

under some (minor) assumptions on the data.

The truncated neural tangent hierarchy at $p$, approximates the dynamic of the finite width neural network upto a time, $t$. This approximation is better and is valid for a longer time for a larger value of $m$ and a larger value of $p$.


**The conjecture**:

This slide from the author's presentation states their conjecture: _Truncated NTH generalizes better with increasing $p$_.

{% include image.html url="https://i.ibb.co/tsPmgGT/11.png" description=" " width="55%" %}  

**Summary**

For finite width neural networks, the gradient dynamic is captured by an infinite hierarchy of recursive differential equations. Truncating these equations to two (_p=2_), we get the Neural Tangent Kernel in the infinite width limit. However, to accurately represent the training dynamics _over the entire duration of training_ of finite width networks we need the entire infinite hierarchy of equations. For feasibility,  truncating this hierarchy to _p_ equations allows us to approximate the training dynamic to certain time, _t_.  Increasing the width, _m_ and/or increasing _p_, allows us to approximate the training dynamics for a longer time. 


## [Proving the lottery ticket hypothesis: Pruning is all you need](https://proceedings.icml.cc/static/paper_files/icml/2020/2313-Paper.pdf)

As the title suggests, this paper provides a proof for the _Lottery Ticket Hypothesis_ (LTH). Though most of the content in the paper is the proof, I won't talk a lot about it. The LTH is fascinating and I try to highlight the significance of actually proving it. But first, I define some terms used in the context of this paper.

_Neural net_: 

<center> $N(x) = W_{L} \sigma(W_{L-1} \sigma(\cdots W_{2} \sigma(W_{1} x)))$ </center>

_Subnetwork_: 

<center> $n(x) = \tilde{W}_{L} \sigma(\tilde{W}_{L-1} \sigma (\cdots \tilde{W}_{2} \sigma(\tilde{W}_{1} x))) \quad \textrm{where} \quad \tilde{W}_{l} = B_{l} \odot W_{l} $ </center> <br/>


Here, $B_{l}$ is a binary matrix of the same size as $W_{l}$ and $\sigma$ denotes the ReLU activation function.


 **(Weak) Lottery ticket Hypothesis**: Consider a randomly initialized network, $N$. $N$ contains a subnetwork $n$ such that when $n$ is trained in isolation it achieves the same performance as (trained) $N$ using at most the same number of iterations.

But, starting with $n$ is entering like entering the lottery with just one ticket. There's a very small chance (close to $0$) that $n$ has the right initialization of parameters to achieve the same performance. A bigger network $N$ contains (exponentially) many small subnetworks, $n$ (many random initializations *i.e.* many lottery tickets) which ensures that you are achieve a good performance when trained. 
 
 **(Strong) Lottery ticket Hypothesis**:  Let $F$ be a fixed target network and let $N$ be a network obtained by overparametrizing $F$. When $N$ is randomly initialized, it contains a subnetwork, $n$, that performs as good as the target network $F$. 

{% include image.html url="https://i.ibb.co/LPvkHRp/lth-pruning-1.png" description="Strong LTH" width="45%" %}  

The reason this is the _Strong_ LTH is that this version states that _you don't need to train the subnetwork at all_. However, it's important to understand that it does not make any explicit claims on the size of the bigger network, $N$.  

Moreover, I found the nomenclature to be a bit confusing. In mathematics, if you assume a stronger version of a hypothesis, the weaker version can be easily proved. This may not be true for LTH. It is definitely not obvious. 
<!---For instance, whenever you're not able to find the subnetwork, $n$, you could just say that the network, $N$ isn't big enough.--->

Denote the width of the network, $F$, by $w$ and it's depth by $d$. The authors prove the strong LTH for $N$ with width, $W = \textrm{polynomial}(w, d)$, and depth, $D = 2d$. Their proof method relies on pruning. Essentially, you can prune $N$ to obtain $n$ which approximates $F$. 

So, if I know that a 10-layer network with width 100 would be sufficient from achieving good performance for a task, can I randomly initialize a network, $N$ with depth 2*10 and width $polynomial(100, 10)$ and prune it to obtain the desired subnetwork? Well, you could. But pruning the network, $N$ would be computationally as hard as training it.

## [Linear Mode Connectivity and the Lottery Ticket Hypothesis](https://proceedings.icml.cc/static/paper_files/icml/2020/5787-Paper.pdf)

Consider a randomly initialized network. Start the optimization with different SGD seeds, *i.e.* a different shuffling of training dataset with  possibly different data augmentation (angle of rotation, flips, etc.). How are the end results of the optimization processes (the trained weights) related to each other? Before answering the question, let's define some terms.

### Linear mode connectivity
Two different solutions, $W_1$ and $W_2$ are said to be linear mode connected if error along the linear path between them doesn't increase. 

Concretely, let $W_\alpha = \alpha W_1 + (1-\alpha) W_2$ where $\alpha \in [0, 1]$. Let $\mathcal{E}(W)$ denote the (train/test) loss where $W$ represents the weights of the network. 

Instability index, $\mathcal{N}$ is defined as $\displaystyle \sup_{\alpha \in [0, 1]} [ \mathcal{E}(W_\alpha) - \mathcal{E}(0.5(W_1 + W_2) ]$.

$W_1$ and $W_2$ are said to be linear mode connected when instability index is close to zero, _i.e._ $\mathcal{N} \approx 0$. 

A network is said to be _SGD noise stable_ if two different runs of SGD (different SGD noises) result in solutions that are linear mode connected. 

The authors show that small networks like LeNet are SGD noise stable _at initialization_. Varying SGD noises with the same initialization results in solutions that are linear mode connected. Larger networks aren't necessarily noise stable at initialization. 

{% include image.html url="https://i.ibb.co/rbw8ybm/lmc-1.png" description=" " width="100%" %}  


However, these bigger networks quickly become stable to SGD noise once they're trained for a few thousand iterations. For ResNet-20 on CIFAR-10, stability occurs at 3% of the total training time, while ResNet-50 on ImageNet becomes stable at 20% of the training process. 

{% include image.html url="https://i.ibb.co/yY4wvTk/lmc-2.png" description="" width="100%" %}  

### Stability and Pruning LTH Subnetworks

The authors in their previous work propose a methodology called _Iterative Magnitude Pruning (IMP)_ for finding the subnetwork from the (weak) LTH that achieves the same performance as the larger one. Interestingly, they find that the IMP subnetworks only train to full accuracy when they are stable to SGD noise. Please refer to the paper for more details regarding this observation.

