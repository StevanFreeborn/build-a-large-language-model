"""Model demo for the GPT model."""

import tiktoken
import torch
from config import GPT_CONFIG_124M
from dummy_gpt_model import DummpyGptModel
from layer_norm import LayerNorm
from torch import nn

from chapter_04.example_dnn import ExampleDeepNeuralNetwork
from chapter_04.feed_forward import FeedForward

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
TEXT_1 = "Every effort moves you"
TEXT_2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(TEXT_1)))
batch.append(torch.tensor(tokenizer.encode(TEXT_2)))
batch = torch.stack(batch, dim=0)
print(batch)

torch.manual_seed(123)
model = DummpyGptModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output logits shape:", logits.shape)
print(logits)

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)
variance = out.var(dim=-1, keepdim=True)
print(mean)
print(variance)

out_norm = (out - mean) / torch.sqrt(variance)
mean = out_norm.mean(dim=-1, keepdim=True)
variance = out_norm.var(dim=-1, keepdim=True)
torch.set_printoptions(sci_mode=False)
print(out_norm)
print(mean)
print(variance)

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print(mean)
print(var)

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)


def print_gradients(model_param, x_param):
    """Print gradients of model parameters."""
    output = model_param(x_param)
    target = torch.tensor([[0.0]])
    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()

    for name, param in model_param.named_parameters():
        if param.grad is not None:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1.0, 0.0, -1.0]])

torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
print_gradients(model_without_shortcut, sample_input)

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)
