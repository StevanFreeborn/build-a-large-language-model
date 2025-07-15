"""Model demo for the GPT model."""

import tiktoken
import torch
from torch import nn

from chapter_04.config import GPT_CONFIG_124M
from chapter_04.dummy_gpt_model import DummpyGptModel
from chapter_04.example_dnn import ExampleDeepNeuralNetwork
from chapter_04.feed_forward import FeedForward
from chapter_04.gpt_model import GPTModel
from chapter_04.layer_norm import LayerNorm
from chapter_04.transformer import TransformerBlock

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

torch.manual_seed(123)
x = torch.randn(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)

print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Generate text using our GPT model."""

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


start_context = "Hello, I am"
print("Start context:", start_context)
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval()

out = generate_text_simple(
    model,
    encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze().tolist())
print("Decoded text:", decoded_text)
