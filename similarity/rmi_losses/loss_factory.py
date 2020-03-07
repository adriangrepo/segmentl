# coding=utf-8

# python 2.X, 3.X compatibility
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

#import torch
import torch.nn as nn

from similarity.rmi_losses import normal_loss
from similarity.rmi_losses import pyramid_loss
from similarity.rmi_losses import rmi
from similarity.rmi_losses.affinity import aaf

def criterion_choose(num_classes=21,
						loss_type='softmax_ce',
						weight=None,
						ignore_index=255,
						reduction='mean',
						max_iter=30000,
						args=None):
	"""choose the criterion to use"""
	info_dict = {
			'softmax_ce': "Normal Softmax Cross Entropy Loss",
			'sigmoid_ce': "Normal Sigmoid Cross Entropy Loss",
			'rmi': "Region Mutual Information Loss",
			'affinity': "Affinity field Loss",
			'pyramid': "Pyramid Loss"
	}
	print("INFO:PyTorch: Using {}.".format(info_dict[loss_type]))
	if loss_type == 'softmax_ce':
		return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
	elif loss_type == 'sigmoid_ce':
		return normal_loss.BCECrossEntropyLoss(num_classes=num_classes, ignore_index=ignore_index)
	elif loss_type == 'rmi':
		return rmi.RMILoss(num_classes=num_classes,
							rmi_radius=args.rmi_radius,
							rmi_pool_way=args.rmi_pool_way,
							rmi_pool_size=args.rmi_pool_size,
							rmi_pool_stride=args.rmi_pool_stride,
							loss_weight_lambda=args.loss_weight_lambda)
	elif loss_type == 'affinity':
		return aaf.AffinityLoss(num_classes=num_classes,
								init_step=args.init_global_step,
								max_iter=max_iter)
	elif loss_type == 'pyramid':
		return pyramid_loss.PyramidLoss(num_classes=num_classes, ignore_index=ignore_index)

	else:
		raise NotImplementedError("The loss type {} is not implemented.".format(loss_type))
