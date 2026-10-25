# Cross-Entropy Loss in Multi-Class Classification

Cross-entropy loss measures the divergence between true label probability distribution $y$ and predicted probability distribution $\hat{y}$.
For multi-class classification with $C$ classes, categorical cross-entropy loss is defined as:
$$L = -\sum_{c=1}^C y_c \log(\hat{y}_c)$$
where $\hat{y}_c = \frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}}$ is the softmax activation over logits $z$.

Combining softmax and cross-entropy yields a remarkably elegant and numerically stable gradient:
$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$
which directly equates to the prediction error on logit $i$.