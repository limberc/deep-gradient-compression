import torch
import torch.distributed.rpc as rpc
import torch.optim as optim


class DGC_SGD(optim.SGD):
    def __init__(self, params, lr=0.001, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, compress_ratio=0.01):
        super(DGC_SGD, self).__init__(params, lr, momentum, dampening,
                                      weight_decay, nesterov)
        self.compress_ratio = compress_ratio
        self.momentum = momentum
        self.nesterov = nesterov
        self.grads_accumulator = {}
        self.momentum_correction = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data

                # Add gradient to the accumulator
                if param in self.grads_accumulator:
                    self.grads_accumulator[param] += grad
                else:
                    self.grads_accumulator[param] = grad.clone()

                # Apply DGC compression
                if self.compress_ratio > 0:
                    self.compress_gradients(param)

                # Update the parameters using SGD with momentum correction
                if self.momentum != 0:
                    param_state = self.state[param]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(param.data)
                        buf.mul_(self.momentum).add_(grad)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(self.momentum).add_(grad)
                    if self.nesterov:
                        grad = grad.add(self.momentum, buf)
                    else:
                        grad = buf

                    # Apply momentum correction for DGC
                    if param in self.momentum_correction:
                        momentum_correction = self.momentum_correction[param]
                        grad.add_(-momentum_correction)

                    # Update momentum correction for DGC
                    momentum_correction = grad.clone()
                    momentum_correction.mul_(self.momentum)
                    self.momentum_correction[param] = momentum_correction

                param.data.add_(-group['lr'], grad)

        return loss

    def compress_gradients(self, param):
        grad = self.grads_accumulator[param]
        numel = grad.numel()

        # Determine the threshold value for sparsification
        k = int(numel * self.compress_ratio)
        if k == 0:
            return

        # Sparsify gradients
        _, indices = torch.topk(grad.abs().view(-1), k)
        mask = torch.zeros_like(grad)
        mask.view(-1).index_fill_(0, indices, 1)
        grad.mul_(mask)

        # Quantize gradients
        grad.div_(mask.sum())
        grad_rounded = grad.round()
        grad_quantized = grad_rounded / grad.numel()

        # Broadcast the compressed gradients to all workers
        rpc.rpc_sync('worker', self._update_gradients, args=(grad_quantized,))

        # Update the gradients accumulator with compressed gradients
        self.grads_accumulator[param] = grad_quantized

        # Clear the accumulator if all workers have updated the gradients
        if rpc.get_worker_info().id == 0:
            self.grads_accumulator[param] = torch.zeros_like(grad)

    def _update_gradients(self, grad_quantized):
        param = next(self.param_groups[0]['params'])
        param.grad.data = grad_quantized
