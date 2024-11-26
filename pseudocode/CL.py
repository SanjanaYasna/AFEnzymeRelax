import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch
import torch.distributed as dist
from torch_geometric.utils import scatter

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
    
#BORROWED DIRECTLY FROM HEAL TRANSFORMER
#this CL loss that is output from a given graph and its perturbed version is added to existing bce classification loss
class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N . 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

#simpler solution: LOSS BASED SOFTMAX
#convert logits to exponential space 
#then you perform softmax 

#OR infomax =informaiton dependent softmax 
class OutputPred(nn.Module):
    def __init__(self, hidden_dim, num_classes = 8):
        super(OutputPred, self).__init__()
        #flatten layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.soft = torch.nn.Softmax(dim=1)
        self.softmax_temp = self.softmax_temp
    #"temperature" normalized softmax 
    def softmax_temp(self, input, t=1.0):
        print("input", input)
        ex = torch.exp(input/t)
        print("exp", ex)
        sum = torch.sum(ex, axis=0)
        return ex / sum
    #by default, it is perturbed
    def forward(self, x, batch, perturbed = True):
        x = self.fc1(x)
        if perturbed:
            random_noise = torch.rand_like(x).to(x.device)
            x2 = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
            #hacky, but batch together samples quickly. Not ideal at all...
            x_batched = scatter(x, batch, dim=0)
            pred = self.fc2(x_batched)
            #mask any values that are 0
            pred = self.soft(torch.exp(pred))
            #now put x and x2 in NT_Xent CL loss!
            return x, x2, pred
        else: 
            x_batched = scatter(x, batch, dim=0)
            pred = self.softmax_temp(self.fc2(x_batched))
            return pred