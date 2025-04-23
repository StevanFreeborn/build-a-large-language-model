import tiktoken
import torch
from config import GPT_CONFIG_124M
from dummy_gpt_model import DummpyGptModel
from torch import nn

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
text1 = "Every effort moves you"
text2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))
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
print(mean)
print(variance)
