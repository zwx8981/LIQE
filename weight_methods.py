from abc import abstractmethod

import numpy as np
import torch
from torch import nn

def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

class WeightingMethod:

    @abstractmethod
    def backward(self, losses, *args, **kwargs):
        pass


class GradCosine(WeightingMethod):
    """Implementation of the unweighted version of the alg. in 'Adapting Auxiliary Losses Using Gradient Similarity'
    """

    def __init__(self, main_task, **kwargs):
        self.main_task = main_task
        self.cosine_similarity = nn.CosineSimilarity(dim=0)

    @staticmethod
    def _flattening(grad):
        return torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(grad)), axis=0)

    def get_grad_cos_sim(self, grad1, grad2):
        """Computes cosine similarity of gradients after flattening of tensors.
        """

        flat_grad1 = self._flattening(grad1)
        flat_grad2 = self._flattening(grad2)

        cosine = nn.CosineSimilarity(dim=0)(flat_grad1, flat_grad2)

        return torch.clamp(cosine, -1, 1)

    def get_grad(self, losses, shared_parameters):
        """
        :param losses: Tensor of losses of shape (n_tasks, )
        :param shared_parameters: model that are not task-specific parameters
        :return:
        """

        main_loss = losses[self.main_task]
        aux_losses = torch.stack(tuple(l for i, l in enumerate(losses) if i != self.main_task))

        main_grad = torch.autograd.grad(main_loss, shared_parameters, retain_graph=True)
        # copy
        grad = tuple(g.clone() for g in main_grad)

        for loss in aux_losses:
            aux_grad = torch.autograd.grad(loss, shared_parameters, retain_graph=True)
            cosine = self.get_grad_cos_sim(main_grad, aux_grad)

            if cosine > 0:
                grad = tuple(g + ga for g, ga in zip(grad, aux_grad))

        return grad

    def backward(self, losses, shared_parameters, returns=True, **kwargs):
        shared_grad = self.get_grad(
            losses,
            shared_parameters=shared_parameters
        )
        loss = torch.sum(torch.stack(losses))
        loss.backward()
        # update grads for shared weights
        for p, g in zip(shared_parameters, shared_grad):
            p.grad = g

        if returns:
            return loss


class GradNorm(WeightingMethod):
    """Implementation of 'GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks'.
    Minor modifications of https://github.com/choltz95/MTGP-NN/blob/master/models.py#L80-L112. See also
    https://github.com/hosseinshn/GradNorm/blob/master/GradNormv10.ipynb
    """
    def __init__(self, n_tasks, alpha=1.5, device=None, **kwargs):
        """
        :param n_tasks:
        :param alpha: the default 1.5 is the same as in the paper for NYU experiments
        """
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.weights = torch.ones((n_tasks, ), requires_grad=True, device=device)
        self.init_losses = None

    def backward(self, losses, last_shared_params, returns=True, **kwargs):
        """Update gradients of the weights.
        :param losses:
        :param last_shared_params:
        :param returns:
        :return:
        """
        if isinstance(losses, list):
            losses = torch.stack(losses)

        if self.init_losses is None:
            self.init_losses = losses.detach().data

        weighted_losses = self.weights * losses
        total_weighted_loss = weighted_losses.sum()
        # compute and retain gradients
        total_weighted_loss.backward(retain_graph=True)
        # zero the w_i(t) gradients since we want to update the weights using gradnorm loss
        self.weights.grad = 0.0 * self.weights.grad

        # compute grad norms
        norms = []
        for w_i, L_i in zip(self.weights, losses):
            dlidW = torch.autograd.grad(L_i, last_shared_params, retain_graph=True)[0]
            norms.append(torch.norm(w_i * dlidW))

        norms = torch.stack(norms)

        # compute the constant term without accumulating gradients
        # as it should stay constant during back-propagation
        with torch.no_grad():
            # loss ratios
            loss_ratios = losses / self.init_losses
            # inverse training rate r(t)
            inverse_train_rates = loss_ratios / loss_ratios.mean()
            constant_term = norms.mean() * (inverse_train_rates ** self.alpha)

        grad_norm_loss = (norms - constant_term).abs().sum()
        self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]

        # make sure sum_i w_i = T, where T is the number of tasks
        with torch.no_grad():
            renormalize_coeff = self.n_tasks / self.weights.sum()
            self.weights *= renormalize_coeff

        if returns:
            return total_weighted_loss


class STL(WeightingMethod):
    """Single task learning
    """

    def __init__(self, main_task, **kwargs):
        self.main_task = main_task

    def backward(self, losses, returns=True, **kwargs):
        loss = losses[self.main_task]
        loss.backward()

        if returns:
            return loss


class Uncertainty(WeightingMethod):
    """For `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics`
    """
    def __init__(self, **kwargs):
        pass

    def backward(self, losses, logsigmas, returns=True, **kwargs):
        loss = sum(
            [1 / (2 * torch.exp(logsigma)) * loss + logsigma / 2 for loss, logsigma in zip(losses, logsigmas)]
        )
        loss.backward()

        if returns:
            return loss


class DynamicWeightAverage(WeightingMethod):
    """Dynamic Weight Average from `End-to-End Multi-Task Learning with Attention`.
    Source: https://github.com/lorenmt/mtan/blob/master/im2im_pred/model_segnet_split.py#L242
    """
    def __init__(self, n_tasks,  n_epochs, n_train_batch, temp=2., **kwargs):
        self.n_tasks = n_tasks
        self.temp = temp
        self.avg_cost = np.zeros([n_epochs, n_tasks], dtype=np.float32)
        self.lambda_weight = np.ones([n_tasks, n_epochs])
        self.n_train_batch = n_train_batch

    def backward(self, losses, epoch, returns=True, **kwargs):
        cost = np.array([detach_to_numpy(l) for l in losses])
        self.avg_cost[epoch, :] += cost / self.n_train_batch

        if epoch == 0 or epoch == 1:
            self.lambda_weight[:, epoch] = 1.0

        else:
            ws = [
                self.avg_cost[epoch - 1, i] / self.avg_cost[epoch - 2, i]
                for i in range(self.n_tasks)
            ]

            for i in range(self.n_tasks):
                self.lambda_weight[i, epoch] = self.n_tasks * np.exp(ws[i] / self.temp) /\
                                               np.sum((np.exp(w / self.temp) for w in ws))

        loss = torch.mean(sum(self.lambda_weight[i, epoch] * losses[i] for i in range(self.n_tasks)))
        loss.backward()

        if returns:
            return loss


class Equal(WeightingMethod):

    def __init__(self, **kwargs):
        pass

    def backward(self, losses, returns=True, **kwargs):
        loss = torch.sum(torch.stack(losses))
        loss.backward()
        if returns:
            return loss


class WeightMethods:

    def __init__(self, method: str, **kwargs):
        """
        :param method:
        """
        baselines = dict(
            stl=STL,
            equal=Equal,
            dwa=DynamicWeightAverage,
            cosine=GradCosine,
            gradnorm=GradNorm,
            uncert=Uncertainty

        )
        assert method in list(baselines.keys()), 'unknown weight method'

        self.method = baselines[method](**kwargs)

    def backwards(self, losses, **kwargs):
        return self.method.backward(losses, **kwargs)
