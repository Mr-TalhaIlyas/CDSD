#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDisentanglementBlock(nn.Module):
    def __init__(self, C, proj_dim):
        super().__init__()
        # Spatial branch weights
        self.W_sc = nn.Linear(C, C//2, bias=True)
        self.bn_sc = nn.BatchNorm1d(C//2)
        # Temporal branch weights
        self.W_tc = nn.Linear(C, C//2, bias=True)
        self.bn_tc = nn.BatchNorm1d(C//2)
        # Final projection
        self.proj = nn.Linear(C, proj_dim, bias=True)
        
        if C == proj_dim:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Linear(C, proj_dim, bias=False)
        
    
    def forward(self, fe):
        # fe: (N, C, T, V)
        N, C, T, V = fe.shape
        
        # Spatial branch: AvgPool along temporal axis → shape (N, C, V)
        fs = fe.mean(dim=2)  # (N, C, V)
        # We want shape (N, V, C) for linear across channels maybe -> transpose
        fs = fs.permute(0, 2, 1)  # (N, V, C)
        fs = self.W_sc(fs)       # (N, V, C)
        # BatchNorm1d expects (N, C, L) so transpose back
        fs = fs.permute(0, 2, 1)  # (N, C, V)
        fs = self.bn_sc(fs)      # (N, C, V)
        fs = F.relu(fs)          # (N, C, V)
        
        # Temporal branch: AvgPool along spatial axis → shape (N, C, T)
        ft = fe.mean(dim=3)      # (N, C, T)
        ft = ft.permute(0, 2, 1)  # (N, T, C)
        ft = self.W_tc(ft)       # (N, T, C)
        ft = ft.permute(0, 2, 1)  # (N, C, T)
        ft = self.bn_tc(ft)      # (N, C, T)
        ft = F.relu(ft)          # (N, C, T)
        
        # Now average pooling each branch to vector:
        vs = fs.mean(dim=2)      # (N, C)
        vt = ft.mean(dim=2)      # (N, C)
        
        # Concatenate:
        v = torch.cat([vs, vt], dim=1)  # (N, 2*C)
        
        # Final projection
        projected_v = self.proj(v)
        skip_v = self.skip(v)
        d_prime = projected_v + skip_v   # (N, proj_dim)
        
        
        return d_prime
 
class MLDecoder(nn.Module):
    """
    ML-Decoder implementation.
    Original paper: "ML-Decoder: Scalable and Top-Performing Multi-Label Image
    Classification with Decoupled Classifiers"
    https://arxiv.org/abs/2111.12933
    """
    def __init__(self, num_classes, initial_num_groups=None, decoder_layers=1,
                 embed_dim=768):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.initial_num_groups = initial_num_groups if initial_num_groups is not None else num_classes
        self.decoder_layers = decoder_layers

        # Learnable group queries
        self.group_queries = nn.Parameter(torch.randn(self.initial_num_groups, embed_dim))

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

        # Per-class classifiers (one for each class)
        self.classifiers = nn.Linear(embed_dim, num_classes)

        # Learnable query-to-class mapping (if using grouping)
        if self.initial_num_groups != self.num_classes:
            self.query_mapping = nn.Linear(self.initial_num_groups, num_classes)
        else:
            self.query_mapping = nn.Identity()

    def forward(self, x):
        # x shape: (batch_size, embed_dim)
        # Add a sequence dimension for the transformer
        x = x.unsqueeze(1)  # -> (batch_size, 1, embed_dim)

        # Expand group queries for the batch
        batch_size = x.shape[0]
        queries = self.group_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Pass through transformer decoder
        # `queries` are the target sequence (what we're refining)
        # `x` is the memory/context (the image features)
        group_features = self.decoder(tgt=queries, memory=x)
        # -> (batch_size, num_groups, embed_dim)

        # Get class logits from the group features
        # Each group's feature vector is passed to all classifiers
        logits = self.classifiers(group_features)
        # -> (batch_size, num_groups, num_classes)

        # The final logit for a class is the max logit across all groups for that class
        # This is the key "decoupling" step
        if self.training:
            # During training, we can use a trick with the query mapping
            # This helps propagate gradients more effectively
            group_logits = logits.mean(dim=2) # Average logits across classes for each group
            mapped_logits = self.query_mapping(group_logits)
            # Combine with max-pooled logits
            max_logits, _ = torch.max(logits, dim=1)
            return mapped_logits + max_logits
        else:
            # During inference, just take the max
            logits, _ = torch.max(logits, dim=1)
            return logits
#%%