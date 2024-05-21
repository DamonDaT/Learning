import torch
import torch.nn as nn

"""
    CASE 1: Multi Head Attention
"""


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_head_size):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads

        # W^Q. Query projection layer.
        self.query_layers = nn.ModuleList([
            nn.Linear(hidden_size, attention_head_size) for _ in range(num_attention_heads)
        ])

        # W^K. Key projection layer.
        self.key_layers = nn.ModuleList([
            nn.Linear(hidden_size, attention_head_size) for _ in range(num_attention_heads)
        ])

        # W^V. Value projection layer.
        self.value_layers = nn.ModuleList([
            nn.Linear(hidden_size, attention_head_size) for _ in range(num_attention_heads)
        ])

    def forward(self, hidden_states):
        # Create a list to store the outputs of each attention head
        all_attention_outputs = []

        for i in range(self.num_attention_heads):
            query_vectors = self.query_layers[i](hidden_states)
            key_vectors = self.key_layers[i](hidden_states)
            value_vectors = self.value_layers[i](hidden_states)

            # Softmax(Q · K ^ T) * V
            attention_scores = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
            attention_scores = torch.softmax(attention_scores, dim=-1)
            attention_outputs = torch.matmul(attention_scores, value_vectors)
            all_attention_outputs.append(attention_outputs)

        return all_attention_outputs


"""
    CASE 2: Multi Query Attention
"""


class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_head_size):
        super(MultiQueryAttention, self).__init__()
        self.num_attention_heads = num_attention_heads

        # W^Q. Query projection layer.
        self.query_layers = nn.ModuleList([
            nn.Linear(hidden_size, attention_head_size) for _ in range(num_attention_heads)
        ])

        # W^K. Key projection layer.
        self.key_layer = nn.Linear(hidden_size, attention_head_size)

        # W^V. Value projection layer.
        self.value_layer = nn.Linear(hidden_size, attention_head_size)

    def forward(self, hidden_states):
        # Create a list to store the outputs of each attention head
        all_attention_outputs = []

        for i in range(self.num_attention_heads):
            query_vectors = self.query_layers[i](hidden_states)
            key_vectors = self.key_layers(hidden_states)
            value_vectors = self.value_layers(hidden_states)

            # Softmax(Q · K ^ T) * V
            attention_scores = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
            attention_scores = torch.softmax(attention_scores, dim=-1)
            attention_outputs = torch.matmul(attention_scores, value_vectors)
            all_attention_outputs.append(attention_outputs)

        return all_attention_outputs


"""
    CASE 3: Grouped Query Attention
"""


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_head_size, num_kv_heads):
        super(GroupedQueryAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads

        # W^Q. Query projection layer.
        self.query_layers = nn.ModuleList([
            nn.Linear(hidden_size, attention_head_size) for _ in range(num_attention_heads)
        ])

        # W^K. Key projection layer.
        self.key_layer = nn.Linear(hidden_size, attention_head_size)

        # W^V. Value projection layer.
        self.value_layer = nn.Linear(hidden_size, attention_head_size)

    def forward(self, hidden_states):
        # Create a list to store the outputs of each attention head
        all_attention_outputs = []

        # The size of each group of kv
        num_queries_per_kv = self.num_attention_heads // self.num_kv_heads

        for i in range(self.num_attention_heads):
            query_vectors = self.query_layers[i](hidden_states)
            # Repeat key vectors
            key_vectors = self.key_layers(hidden_states)
            key_vectors = torch.repeat_interleave(key_vectors, num_queries_per_kv, dim=2)
            # Repeat value vectors
            value_vectors = self.value_layers(hidden_states)
            value_vectors = torch.repeat_interleave(value_vectors, num_queries_per_kv, dim=2)

            # Softmax(Q · K ^ T) * V
            attention_scores = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))
            attention_scores = torch.softmax(attention_scores, dim=-1)
            attention_outputs = torch.matmul(attention_scores, value_vectors)
            all_attention_outputs.append(attention_outputs)

        return all_attention_outputs
