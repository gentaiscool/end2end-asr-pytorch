import torch

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer, min_lr=1e-5):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.min_lr = min_lr

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        "Implement `lrate` above"
        step = self._step
        return max(self.min_lr, self.factor * \
            (self.model_size ** (-0.5) * min(step **
                                             (-0.5), step * self.warmup ** (-1.5))))

class AnnealingOpt:
    "Optim wrapper for annealing opt"

    def __init__(self, lr, lr_anneal, optimizer):
        self.optimizer = optimizer
        self.lr = lr
        self.lr_anneal = lr_anneal
    
    def step(self):
        optim_state = self.optimizer.state_dict()
        optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / self.lr_anneal
        self.optimizer.load_state_dict(optim_state)

# class SGDOpt:
#     "Optim wrapper that implements SGD"

#     def __init__(self, parameters, lr, momentum, nesterov=True):
#         self.optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, nesterov=nesterov)