import torch
import torchvision.models as models
import torch.nn as nn 

import tensorly as tl 
from tensorly.decomposition import partial_tucker 
tl.set_backend('pytorch') 

import GPyOpt 
import GPy 
from GPyOpt.models.gpmodel import GPModel 
from GPyOpt.core.task.space import Design_space
from GPyOpt.acquisitions.EI import AcquisitionEI

import numpy as np

global conv

model = models.vgg16(pretrained=True)


class BayesOpt_rank_selection():
    def f(self, x):
        x1 = x[:,0] 
        x2 = x[:,1] 

        ranks = [int(x1), int(x2)]

        core,[last,first] = partial_tucker(conv.weight.data, modes=[0,1], ranks=ranks, init='svd')

        recon_error = tl.norm(conv.weight.data - tl.tucker_to_tensor((core,[last,first])),2) / tl.norm(conv.weight.data,2) 

        #recon_error = np.nan_to_num(recon_error) 

        ori_out = conv.weight.data.shape[0] 
        ori_in = conv.weight.data.shape[1]
        ori_ker = conv.weight.data.shape[2] 
        ori_ker2 = conv.weight.data.shape[3] 

        first_out = first.shape[0] 
        first_in = first.shape[1]

        core_out = core.shape[0] 
        core_in = core.shape[1]

        last_out = last.shape[0]
        last_in = last.shape[1] 

        original_computation = ori_out*ori_in*ori_ker*ori_ker2
        decomposed_computation = (first_out*first_in) + (core_in*core_out*ori_ker*ori_ker2) + (last_in*last_out)

        computation_error = decomposed_computation/original_computation

        Error = float(recon_error + computation_error) 

        return Error 

def estimate_ranks_BayesOpt():
    
    func = BayesOpt_rank_selection()

    axis_0 = conv.weight.data.shape[0] 
    axis_1 = conv.weight.data.shape[1] 

    space = [{'name':'rank_1', 'type':'continuous', 'domain':(1,axis_0)}, {'name':'rank_2','type':'continuous','domain':(1,axis_1)}]

    feasible_region = GPyOpt.Design_space(space=space)

    initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 10) 

    objective = GPyOpt.core.task.SingleObjective(func.f) 

    model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=10, verbose=False) 

    acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)
    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=acquisition_optimizer) 

    evaluator = GPyOpt.core.evaluators.Sequential(acquisition) 

    bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design) 

    max_time = None
    tolerance = 10e-3 
    max_iter = 30 
    bo.run_optimization(max_iter=max_iter, max_time=max_time, eps=tolerance, verbosity=False) 

    bo.plot_acquisition()
    bo.plot_convergence() 

    rank1 = int(bo.x_opt[0]) 
    rank2 = int(bo.x_opt[1]) 
    ranks = [rank1, rank2] 

    return ranks

def BayesOpt_tucker_decomposition():
    ranks = estimate_ranks_BayesOpt()
    print(conv, "BayesOpt estimated ranks", ranks)
    core, [last, first] = partial_tucker(conv.weight.data, modes=[0,1], tol=10e-5, ranks=ranks, init='svd') 

    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], out_channels=first.shape[1], kernel_size=1, stride=1, padding=0, dilation=conv.dilation, bias=False) 

    core_layer = torch.nn.Conv2d(in_channels=core.shape[1], out_channels=core.shape[0], kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, bias=False) 

    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], out_channels=last.shape[1], kernel_size=1, stride=1, padding=0, dilation=0) 

    first_layer.weight.data = torch.transpose(first,1,0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core 

    new_layers = [first_layer, core_layer, last_layer] 
    return nn.Sequential(*new_layers) 


for i, key in enumerate(model.features._modules.keys()):
    if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
        conv = model.features._modules[key].cuda().eval().cpu()
        decomposed = BayesOpt_tucker_decomposition() 
        model.features._modules[key] = decomposed 
    torch.save(model, 'BayesOpt_decomposed_model') 
