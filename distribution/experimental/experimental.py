import numpy as np
import torch
#from keras import backend as K
#import tensorflow as tf

#after https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels/blob/master/loss.py
def cross_entropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def symmetric_cross_entropy(alpha, beta):
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = torch.clamp(y_pred_1, 1e-7, 1.0)
        y_true_2 = torch.clamp(y_true_2, 1e-4, 1.0)

        #tf Batch x height x width x Channel
        #pytorch  Batch x Channel x height x width
        CE = torch.mean(-torch.sum(y_true_1 * torch.log(y_pred_1), axis=1))
        RCE = torch.mean(-torch.sum(y_pred_2 * torch.log(y_true_2), axis=1))
        return alpha * CE + beta * RCE
        #return alpha*torch.mean(-torch.sum(y_true_1 * torch.log(y_pred_1), axis = -1))
        # + beta*torch.mean(-torch.sum(y_pred_2 * torch.log(y_true_2), axis = -1))
    return loss


def lsr(y_true, y_pred):
    epsilon = 0.1
    y_smoothed_true = y_true * (1 - epsilon - epsilon / 10.0)
    y_smoothed_true = y_smoothed_true + epsilon / 10.0

    y_pred_1 = torch.clamp(y_pred, 1e-7, 1.0)

    return torch.mean(-torch.sum(y_smoothed_true * torch.log(y_pred_1), axis=-1))

def generalized_cross_entropy(y_true, y_pred):
    """
    2018 - nips - Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels.
    """
    q = 0.7
    t_loss = (1 - tf.pow(torch.sum(y_true * y_pred, axis=-1), q)) / q
    return torch.mean(t_loss)

def joint_optimization_loss(y_true, y_pred):
    """
    2018 - cvpr - Joint optimization framework for learning with noisy labels.
    """
    y_pred_avg = K.mean(y_pred, axis=0)
    p = np.ones(10, dtype=np.float32) / 10.
    l_p = - K.sum(K.log(y_pred_avg) * p)
    l_e = K.categorical_crossentropy(y_pred, y_pred)
    return K.categorical_crossentropy(y_true, y_pred) + 1.2 * l_p + 0.8 * l_e

def boot_soft(y_true, y_pred):
    """
    2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
    """
    beta = 0.95

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return -K.sum((beta * y_true + (1. - beta) * y_pred) *
                  K.log(y_pred), axis=-1)


def boot_hard(y_true, y_pred):
    """
    2015 - iclrws - Training deep neural networks on noisy labels with bootstrapping.
    """
    beta = 0.8
    epsilon = 1e-7
    y_pred /= torch.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    pred_labels = K.one_hot(K.argmax(y_pred, 1), num_classes=K.shape(y_true)[1])
    return -K.sum((beta * y_true + (1. - beta) * pred_labels) *
           K.log(y_pred), axis=-1)
