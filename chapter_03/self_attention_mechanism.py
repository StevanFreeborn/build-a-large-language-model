"""A self attention mechanism implemented with PyTorch."""

import torch

from self_attention import SelfAttentionV1, SelfAttentionV2

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]
)

x_2 = inputs[1]
d_in = inputs.shape[1]
D_OUT = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, D_OUT), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, D_OUT), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, D_OUT), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

torch.manual_seed(123)
sa_v1 = SelfAttentionV1(d_in, D_OUT)
print(sa_v1(inputs))

torch.manual_seed(789)
sa_v2 = SelfAttentionV2(d_in, D_OUT)
print(sa_v2(inputs))

sa_v1.W_query = torch.nn.Parameter(sa_v2.w_query.weight.T)
sa_v1.W_key = torch.nn.Parameter(sa_v2.w_key.weight.T)
sa_v1.W_value = torch.nn.Parameter(sa_v2.w_value.weight.T)

print(sa_v1(inputs))
