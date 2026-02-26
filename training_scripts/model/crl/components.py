import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Union, Tuple

THRESHOLD = 0.5
INIT_RANGE = 0.5
EPSILON = 1e-10
INIT_L = 0.0


@dataclass
class Connection:
    prev_layer: Any = None
    is_skip_to_layer: bool = False
    skip_from_layer: Any = None


class GradGraft(torch.autograd.Function):
    """Implement the Gradient Grafting."""

    @staticmethod
    def forward(ctx, X, Y):
        return X

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


class Binarizer(torch.autograd.Function):

    @staticmethod
    def forward(_, concepts):
        hard_concepts = (concepts.detach() > 0.0).float()
        return hard_concepts

    @staticmethod
    def backward(_, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinarizeLayer(nn.Module):
    def __init__(self, n_concepts, use_not):
        super(BinarizeLayer, self).__init__()
        self.n_concepts = n_concepts
        self.use_not = use_not
        self.input_dim = n_concepts
        self.output_dim = 2 * n_concepts if use_not else n_concepts
        self.layer_type = "binarization"
        self.dim2id = {i: i for i in range(self.output_dim)}

    def forward(self, x):
        x = Binarizer.apply(x)
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return x

    @torch.no_grad()
    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        pass

    def get_rule_name(self, concept_names):
        self.rule_name = []
        for i in range(self.n_concepts):
            self.rule_name.append(concept_names[i])
        if self.use_not:
            for i in range(self.n_concepts):
                self.rule_name.append("~" + concept_names[i])


# class BinarizeLayer(nn.Module):
#     def __init__(self, n_concepts, use_not, k=5.0, temperature_decay=True):
#         super().__init__()
#         self.use_not = bool(use_not)
#         self.n_concepts = int(n_concepts)
#         self.input_dim = self.n_concepts
#         self.output_dim = 2 * self.n_concepts if self.use_not else self.n_concepts
#         self.layer_type = "soft_binarization"
#         self.dim2id = {i: i for i in range(self.output_dim)}
        
#         # Temperature scheduling (optional but recommended)
#         self.temperature_decay = temperature_decay
#         if temperature_decay:
#             self.register_buffer('k', torch.tensor(float(k)))
#             self.k_min = 0.5
#         else:
#             self.k = float(k)

#     def forward(self, x):
#         # x: [B, n_concepts] - should be LOGITS or raw scores
#         x32 = x.float()
        
#         # Method 1: STE on raw logits (simpler, more stable)
#         h = (x32.detach() > 0.0).float()  # hard binarization
#         soft_gate = torch.sigmoid(self.k * x32)  # soft approximation for gradients
#         x_bin = h + (soft_gate - soft_gate.detach())  # STE: forward=hard, backward=soft
        
#         if self.use_not:
#             x_bin = torch.cat([x_bin, 1 - x_bin], dim=1)
        
#         return x_bin.to(dtype=x.dtype, device=x.device)

#     @torch.no_grad()
#     def binarized_forward(self, x):
#         """Pure binarization without gradients (for inference/rule extraction)"""
#         x32 = x.float()
#         h = (x32 > 0.0).float()
#         if self.use_not:
#             h = torch.cat([h, 1 - h], dim=1)
#         return h.to(dtype=x.dtype, device=x.device)

#     def clip(self):
#         pass  # No parameters to clip in binarization layer
    
#     def anneal_temperature(self, epoch, total_epochs):
#         """Optional: Decay temperature during training for sharper gradients"""
#         if self.temperature_decay:
#             progress = epoch / max(1, total_epochs)
#             self.k = max(self.k_min, 5.0 * (1 - progress) + self.k_min * progress)

class RMSNorm(nn.Module):
    r"""
    Root Mean Square Layer Normalization (approximate PyTorch 2.9+ behavior).

    Args:
        normalized_shape (int or tuple or torch.Size):
            Shape of the last D dimensions to normalize over.
        eps (float, optional):
            Small term added to the denominator for numerical stability.
            If None, uses torch.finfo(x.dtype).eps at runtime.
        elementwise_affine (bool):
            If True, has learnable per-element weights (gamma).
        device, dtype:
            Passed to parameter initialization (like other nn modules).
    """
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: Optional[float]
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...], torch.Size],
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        elif isinstance(normalized_shape, torch.Size):
            normalized_shape = tuple(normalized_shape)
        elif isinstance(normalized_shape, tuple):
            # already fine
            pass
        else:
            raise TypeError(
                f"normalized_shape must be int, tuple or torch.Size, "
                f"got {type(normalized_shape)}"
            )

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        factory_kwargs = {"device": device, "dtype": dtype}

        if self.elementwise_affine:
            # per-element scale (gamma) over the last D dims
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        # Normalize over the last len(normalized_shape) dimensions
        dims = tuple(range(-len(self.normalized_shape), 0))

        # pick eps based on dtype if not provided
        eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps

        # compute RMS over those dimensions
        # RMS(x) = sqrt( mean(x^2) + eps )
        rms = x.pow(2).mean(dim=dims, keepdim=True).add(eps).sqrt()

        y = x / rms

        if self.weight is not None:
            # weight has shape normalized_shape, broadcast over batch/other dims
            y = y * self.weight

        return y

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class Product(torch.autograd.Function):
    """Tensor product function."""

    @staticmethod
    def forward(ctx, X):
        y = -1.0 / (-1.0 + torch.sum(torch.log(X), dim=1))
        ctx.save_for_backward(X, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        X, y = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + EPSILON))
        return grad_input


class LRLayer(nn.Module):
    """The LR layer is used to learn the linear part of the data."""

    def __init__(self, input_dim, output_dim):
        super(LRLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_type = "linear"
        self.rid2dim = None
        self.rule2weights = None

        self.fc1 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        return self.fc1(x)

    @torch.no_grad()
    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        for param in self.fc1.parameters():
            param.data.clamp_(-1.0, 1.0)

    def l1_norm(self):
        return torch.norm(self.fc1.weight, p=1)

    def l2_norm(self):
        return torch.sum(self.fc1.weight**2)

    def get_rule2weights(self, prev_layer, skip_connect_layer):
        prev_layer = self.conn.prev_layer
        skip_connect_layer = self.conn.skip_from_layer

        always_act_pos = prev_layer.node_activation_cnt == prev_layer.forward_tot
        merged_dim2id = prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
        if skip_connect_layer is not None:
            shifted_dim2id = {
                (k + prev_layer.output_dim): (-2, v)
                for k, v in skip_connect_layer.dim2id.items()
            }
            merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})
            always_act_pos = torch.cat(
                [
                    always_act_pos,
                    (
                        skip_connect_layer.node_activation_cnt
                        == skip_connect_layer.forward_tot
                    ),
                ]
            )

        Wl, bl = list(self.fc1.parameters())
        bl = torch.sum(Wl.T[always_act_pos], dim=0) + bl
        Wl = Wl.cpu().detach().numpy()
        self.bl = bl.cpu().detach().numpy()

        marked = defaultdict(lambda: defaultdict(float))
        rid2dim = {}
        for label_id, wl in enumerate(Wl):
            for i, w in enumerate(wl):
                rid = merged_dim2id[i]
                if rid == -1 or rid[1] == -1:
                    continue
                marked[rid][label_id] += w
                rid2dim[rid] = i % prev_layer.output_dim

        self.rid2dim = rid2dim
        self.rule2weights = sorted(
            marked.items(), key=lambda x: max(map(abs, x[1].values())), reverse=True
        )


class ConjunctionLayer(nn.Module):
    """The conjunction layer is used to learn the conjunction of nodes."""

    def __init__(self, input_dim, output_dim, use_not=False):
        super(ConjunctionLayer, self).__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.layer_type = "conjunction"

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.input_dim, self.output_dim))
        self.node_activation_cnt = None

    def forward(self, x):
        res_tilde = self.continuous_forward(x)
        res_bar = self.binarized_forward(x)
        return GradGraft.apply(res_bar, res_tilde)

    def continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return Product.apply(1 - (1 - x).unsqueeze(-1) * self.W)

    @torch.no_grad()
    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarizer.apply(self.W - THRESHOLD)
        return torch.prod(1 - (1 - x).unsqueeze(-1) * Wb, dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class DisjunctionLayer(nn.Module):
    """The disjunction layer is used to learn the disjunction of nodes."""

    def __init__(self, input_dim, output_dim, use_not=False):
        super(DisjunctionLayer, self).__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.layer_type = "disjunction"

        self.W = nn.Parameter(INIT_RANGE * torch.rand(self.input_dim, self.output_dim))
        self.node_activation_cnt = None

    def forward(self, x):
        res_tilde = self.continuous_forward(x)
        res_bar = self.binarized_forward(x)
        return GradGraft.apply(res_bar, res_tilde)

    def continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return 1 - Product.apply(1 - x.unsqueeze(-1) * self.W)

    @torch.no_grad()
    def binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = Binarizer.apply(self.W - THRESHOLD)
        return 1 - torch.prod(1 - x.unsqueeze(-1) * Wb, dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class UnionLayer(nn.Module):

    def __init__(self, input_dim, output_dim, use_not=False):
        super(UnionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim * 2
        self.use_not = use_not
        self.layer_type = "union"
        self.forward_tot = None
        self.node_activation_cnt = None
        self.dim2id = None
        self.rule_list = None
        self.rule_name = None

        self.con_layer = ConjunctionLayer(self.input_dim, output_dim, use_not=use_not)
        self.dis_layer = DisjunctionLayer(self.input_dim, output_dim, use_not=use_not)

    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)

    def binarized_forward(self, x):
        return torch.cat(
            [self.con_layer.binarized_forward(x), self.dis_layer.binarized_forward(x)],
            dim=1,
        )

    def edge_count(self):
        con_Wb = Binarizer.apply(self.con_layer.W - THRESHOLD)
        dis_Wb = Binarizer.apply(self.dis_layer.W - THRESHOLD)
        return torch.sum(con_Wb) + torch.sum(dis_Wb)

    def l1_norm(self):
        return torch.sum(self.con_layer.W) + torch.sum(self.dis_layer.W)

    def l2_norm(self):
        return torch.sum(self.con_layer.W**2) + torch.sum(self.dis_layer.W**2)

    def clip(self):
        self.con_layer.clip()
        self.dis_layer.clip()

    def get_rules(self, prev_layer, skip_connect_layer):
        self.con_layer.forward_tot = self.dis_layer.forward_tot = self.forward_tot
        self.con_layer.node_activation_cnt = self.dis_layer.node_activation_cnt = (
            self.node_activation_cnt
        )

        # get dim2id and rule lists of the conjunction layer and the disjunction layer
        # dim2id: dimension --> (k, rule id)
        con_dim2id, con_rule_list = extract_rules(
            prev_layer, skip_connect_layer, self.con_layer
        )
        dis_dim2id, dis_rule_list = extract_rules(
            prev_layer, skip_connect_layer, self.dis_layer, self.con_layer.W.shape[1]
        )

        shift = max(con_dim2id.values()) + 1
        dis_dim2id = {k: (-1 if v == -1 else v + shift) for k, v in dis_dim2id.items()}
        dim2id = defaultdict(lambda: -1, {**con_dim2id, **dis_dim2id})

        rule_list = (con_rule_list, dis_rule_list)

        self.dim2id = dim2id
        self.rule_list = rule_list

    def get_rule_description(self, input_rule_name, wrap=False):
        """
        input_rule_name: (skip_connect_rule_name, prev_rule_name)
        """
        self.rule_name = []
        for rl, op in zip(self.rule_list, ("&", "|")):
            for rule in rl:
                name = ""
                for i, ri in enumerate(rule):
                    op_str = " {} ".format(op) if i != 0 else ""
                    layer_shift = ri[0]
                    not_str = ""
                    if ri[0] > 0:  # ri[0] == 1 or ri[0] == 2
                        layer_shift *= -1
                        not_str = "~"
                    var_str = ("({})" if (wrap or not_str == "~") else "{}").format(
                        input_rule_name[2 + layer_shift][ri[1]]
                    )
                    name += op_str + not_str + var_str
                self.rule_name.append(name)


def extract_rules(prev_layer, skip_connect_layer, layer, pos_shift=0):
    # dim2id = {dimension: rule_id} :
    dim2id = defaultdict(lambda: -1)
    rules = {}
    tmp = 0
    rule_list = []

    # Wb.shape = (output_dim, input_dim)
    Wb = (layer.W.t() > 0.5).type(torch.int).detach().cpu().numpy()

    # merged_dim2id is the dim2id of the input (the prev_layer and skip_connect_layer)
    merged_dim2id = prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
    if skip_connect_layer is not None:
        shifted_dim2id = {
            (k + prev_layer.output_dim): (-2, v)
            for k, v in skip_connect_layer.dim2id.items()
        }
        merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})

    for ri, row in enumerate(Wb):
        # delete dead nodes
        no_activated = layer.node_activation_cnt[ri + pos_shift] == 0
        all_activated = layer.node_activation_cnt[ri + pos_shift] == layer.forward_tot
        if no_activated or all_activated:
            dim2id[ri + pos_shift] = -1
            continue

        # rule[i] = (k, rule_id):
        #     k == -1: connects to a rule in prev_layer,
        #     k ==  1: connects to a rule in prev_layer (NOT),
        #     k == -2: connects to a rule in skip_connect_layer,
        #     k ==  2: connects to a rule in skip_connect_layer (NOT).
        rule = {}
        for i, w in enumerate(row):
            # deal with "use NOT", use_not_mul = -1 if it used NOT in that input dimension
            use_not_mul = 1
            if layer.use_not:
                if i >= layer.input_dim // 2:
                    use_not_mul = -1
                i = i % (layer.input_dim // 2)

            if w > 0 and merged_dim2id[i][1] != -1:
                rid = merged_dim2id[i]
                rule[(rid[0] * use_not_mul, rid[1])] = 1

        # give each unique rule an id, and save this id in dim2id
        rule = tuple(sorted(rule.keys()))
        if rule not in rules:
            rules[rule] = tmp
            rule_list.append(rule)
            dim2id[ri + pos_shift] = tmp
            tmp += 1
        else:
            dim2id[ri + pos_shift] = rules[rule]
    return dim2id, rule_list

class ScaledSoftsign(nn.Module):
    """Softsign with adjustable bounds - gentlest saturation, best gradient preservation"""
    def __init__(self, scale=3.0, beta=1.0):
        super().__init__()
        self.scale = scale
        self.beta = beta
    
    def forward(self, x):
        return self.scale * x / (self.beta + torch.abs(x))

#%% ORIGINAL VERSION WITHOUT L1 AND ENTROPY REGULARIZATION
class ConceptLogicLayers(nn.Module):
    """
    Concept Reasoning Layer (CRL) for action recognition.
    Learns logical rules from concepts to predict action classes.
    """
    def __init__(self, n_concepts, n_actions, l1_dim=256, l2_dim=256, 
                 use_not=True, use_skip=True, temperature=1.0, l2_weight=5e-6):
        super(ConceptLogicLayers, self).__init__()

        self.n_concepts = n_concepts
        self.n_actions = n_actions
        self.l1_dim = l1_dim
        self.l2_dim = l2_dim
        self.use_not = use_not
        self.use_skip = use_skip
        self.l2_weight = l2_weight
        self.temperature = temperature

        # self.tanh = nn.Tanh()
        self.scaled_softsign = ScaledSoftsign()
        # self.norm = nn.LayerNorm(n_concepts)
        self.norm = RMSNorm(n_concepts)
        
        self.layer_list = nn.ModuleList()
        self.dim_list = [self.n_concepts, self.l1_dim, self.l2_dim, self.n_actions]

        prev_layer_dim = None

        for idx, dim in enumerate(self.dim_list):
            # Skip connections (from original CRL)
            skip_from_layer = None
            if self.use_skip and idx >= 3:
                skip_from_layer = self.layer_list[-2]
                prev_layer_dim += skip_from_layer.output_dim
            
            # Create layer (MAINTAINING ORIGINAL CRL LOGIC)
            if idx == 0:
                # First layer: Binarization
                layer = BinarizeLayer(dim, self.use_not)
                layer_name = f"binary{idx}"
            elif idx == len(self.dim_list) - 1:
                # Last layer: Linear classification
                layer = LRLayer(prev_layer_dim, dim)
                layer_name = f"lr{idx}"
            else:
                # Middle layers: Union (conjunction + disjunction)
                # NOTE: First logical layer does NOT use NOT if binarization already used it
                layer_use_not = True if idx != 1 else False
                layer = UnionLayer(prev_layer_dim, dim, use_not=layer_use_not)
                layer_name = f"union{idx}"
            
            # Set connections (original CRL connection logic)
            layer.conn = Connection(
                prev_layer= self.layer_list[-1] if len(self.layer_list) > 0 else None,
                is_skip_to_layer=False,
                skip_from_layer=skip_from_layer
            )
            
            if skip_from_layer is not None:
                skip_from_layer.conn.is_skip_to_layer = True
            
            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)
            
        # Temperature parameter (from original CRL)
        self.t = nn.Parameter(torch.log(torch.tensor([self.temperature])))
        
    def forward(self, concepts):
        concepts = self.norm(concepts)
        concepts = self.scaled_softsign(concepts)#* 5.0  # Scale tanh output from [-1, 1] to [-2, 2]
        x = concepts

        for layer in self.layer_list:
            if layer.conn.skip_from_layer is not None:
                x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
                del layer.conn.skip_from_layer.x_res
            x = layer(x)
            if layer.conn.is_skip_to_layer:
                layer.x_res = x
        # Apply temperature scaling
        x = x / torch.exp(self.t)
        
        return x
    
    def l2_penalty(self):
        """Compute L2 penalty on logic layers (excluding binarization)."""
        l2_penalty = 0.0
        for layer in self.layer_list[1:]:  # Skip binarization layer
            if hasattr(layer, 'l2_norm'):
                l2_penalty += layer.l2_norm()
        return l2_penalty
    
    def clip_weights(self):
        """Clip weights to [0, 1] for logic layers."""
        for layer in self.layer_list[:-1]: # [:-1]  # Exclude last linear layer
            if hasattr(layer, 'clip'):
                layer.clip()


#%% NEW VERSION WITH L1 AND ENTROPY REGULARIZATION


# class ConceptLogicLayers(nn.Module):
#     """
#     Concept Reasoning Layer (CRL) for action recognition.
#     Learns logical rules from concepts to predict action classes.
#     """
#     def __init__(self, n_concepts, n_actions, l1_dim=256, l2_dim=256, 
#                  use_not=True, use_skip=True, temperature=1.0, l2_weight=5e-6):
#         super(ConceptLogicLayers, self).__init__()

#         self.n_concepts = n_concepts
#         self.n_actions = n_actions
#         self.l1_dim = l1_dim
#         self.l2_dim = l2_dim
#         self.use_not = use_not
#         self.use_skip = use_skip
#         self.l2_weight = l2_weight
#         self.l1_weight = 1e-8          # NEW
#         self.entropy_weight = 1e-4 # NEW
#         self.temperature = temperature

#         # NEW: Layer-specific regularization weights (decay for deeper layers)
#         # We have 3 trainable layers usually: Union1, Union2, LinearHead
#         # self.layer_reg_weights = [1.0, 0.8, 0.5] 
#         self.layer_reg_weights = [1.0, 1.0, 1.0] 
#         self.norm = RMSNorm(n_concepts)
#         self.scaled_softsign = ScaledSoftsign()
        
#         self.layer_list = nn.ModuleList()
#         self.dim_list = [self.n_concepts, self.l1_dim, self.l2_dim, self.n_actions]

#         prev_layer_dim = None

#         for idx, dim in enumerate(self.dim_list):
#             # Skip connections (from original CRL)
#             skip_from_layer = None
#             if self.use_skip and idx >= 3:
#                 skip_from_layer = self.layer_list[-2]
#                 prev_layer_dim += skip_from_layer.output_dim
            
#             # Create layer (MAINTAINING ORIGINAL CRL LOGIC)
#             if idx == 0:
#                 # First layer: Binarization
#                 layer = BinarizeLayer(dim, self.use_not)
#                 layer_name = f"binary{idx}"
#             elif idx == len(self.dim_list) - 1:
#                 # Last layer: Linear classification
#                 layer = LRLayer(prev_layer_dim, dim)
#                 layer_name = f"lr{idx}"
#             else:
#                 # Middle layers: Union (conjunction + disjunction)
#                 # NOTE: First logical layer does NOT use NOT if binarization already used it
#                 layer_use_not = True if idx != 1 else False
#                 layer = UnionLayer(prev_layer_dim, dim, use_not=layer_use_not)
#                 layer_name = f"union{idx}"
            
#             # Set connections (original CRL connection logic)
#             layer.conn = Connection(
#                 prev_layer= self.layer_list[-1] if len(self.layer_list) > 0 else None,
#                 is_skip_to_layer=False,
#                 skip_from_layer=skip_from_layer
#             )
            
#             if skip_from_layer is not None:
#                 skip_from_layer.conn.is_skip_to_layer = True
            
#             prev_layer_dim = layer.output_dim
#             self.add_module(layer_name, layer)
#             self.layer_list.append(layer)
            
#         # Temperature parameter (from original CRL)
#         self.t = nn.Parameter(torch.log(torch.tensor([self.temperature])))
        
#     def forward(self, concepts):
#         concepts = self.norm(concepts)
#         concepts = self.scaled_softsign(concepts)# * 5.0  # Scale tanh output from [-1, 1] to [-2, 2]
#         x = concepts

#         for layer in self.layer_list:
#             if layer.conn.skip_from_layer is not None:
#                 x = torch.cat((x, layer.conn.skip_from_layer.x_res), dim=1)
#                 del layer.conn.skip_from_layer.x_res
#             x = layer(x)
#             if layer.conn.is_skip_to_layer:
#                 layer.x_res = x
#         # Apply temperature scaling
#         x = x / torch.exp(self.t)
        
#         return x
    
#     def l2_penalty(self):
#         """Compute layer-weighted L2 penalty on logic layers."""
#         l2_penalty = 0.0
#         layer_idx = 0
#         # layer_list[0] is BinarizeLayer (no params), so we skip it
#         for layer in self.layer_list[1:]: 
#             if hasattr(layer, 'l2_norm'):
#                 # Apply layer-specific weight safely
#                 w_idx = min(layer_idx, len(self.layer_reg_weights)-1)
#                 weight = self.layer_reg_weights[w_idx]
#                 l2_penalty += weight * layer.l2_norm()
#                 layer_idx += 1
#         return l2_penalty
    
#     def l1_penalty(self):
#         """NEW: Compute L1 penalty for sparsity (interpretable rules)."""
#         l1_penalty = 0.0
#         # Exclude BinarizeLayer (idx 0) and usually exclude Final Linear (last idx) 
#         # if we only want logic sparsity, but applying to all trainable layers is also valid.
#         # Here we apply to UnionLayers (logic) specifically.
#         for layer in self.layer_list[1:-1]: 
#             if hasattr(layer, 'l1_norm'):
#                 l1_penalty += layer.l1_norm()
#         return l1_penalty
    
#     def entropy_penalty(self):
#         """NEW: Encourage decisive weights (either 0 or 1) for interpretability."""
#         entropy_loss = 0.0
#         for layer in self.layer_list[1:-1]:
#             if isinstance(layer, UnionLayer):
#                 # Entropy of weights (penalize uncertain weights around 0.5)
#                 # We check both Conjunction and Disjunction parts
#                 for sub_layer in [layer.con_layer, layer.dis_layer]:
#                     W = sub_layer.W
#                     # Clamp to avoid log(0)
#                     p = torch.clamp(W, 1e-7, 1.0 - 1e-7)
#                     entropy = -p * torch.log(p) - (1-p) * torch.log(1-p)
#                     entropy_loss += entropy.mean()
#         return entropy_loss
    
#     def combined_regularization(self, epoch, total_epochs, warmup_epochs):
#         """
#         NEW: Adaptive regularization with explicit warmup handling.
#         """
#         # 1. Warmup Phase: Pure L2 (Stability)
#         if epoch < warmup_epochs:
#             return self.l2_weight * self.l2_penalty()

#         # 2. Curriculum Phase
#         # Adjust progress to start AFTER warmup
#         effective_epoch = epoch - warmup_epochs
#         effective_total = max(1, total_epochs - warmup_epochs)
#         progress = effective_epoch / effective_total
        
#         # L2: Strong early, decay later
#         l2_scale = max(0.3, 1.0 - 0.5 * progress)
#         l2_loss = l2_scale * self.l2_penalty()
        
#         # L1: Ramp up gradually (Sparsity)
#         # Start immediately after warmup, peak at 50% of remaining time
#         if progress < 0.4:
#             l1_scale = progress / 0.4
#         else:
#             l1_scale = 1.0
#         l1_loss = l1_scale * self.l1_penalty()
        
#         # Entropy: Late training (Hardening)
#         # Start at 50% of remaining time
#         entropy_scale = max(0.0, (progress - 0.5) * 2.0)
#         entropy_loss = entropy_scale * self.entropy_penalty()
        
#         return (self.l2_weight * l2_loss + 
#                 self.l1_weight * l1_loss + 
#                 self.entropy_weight * entropy_loss)

#     def clip_weights(self):
#         """Clip weights to [0, 1] for logic layers."""
#         for layer in self.layer_list[:-1]: # [:-1]  # Exclude last linear layer
#             if hasattr(layer, 'clip'):
#                 layer.clip()
                
#     def orthogonal_penalty(self):
#         """Encourage diverse logical rules by penalizing correlated weights."""
#           #orthogonal regularization to encourage diverse logical rules
#         ortho_loss = 0.0
#         for layer in self.layer_list[1:-1]:
#             if isinstance(layer, UnionLayer):
#                 for W in [layer.con_layer.W, layer.dis_layer.W]:
#                     # W: [input_dim, output_dim]
#                     W_norm = F.normalize(W, p=2, dim=0)  # Normalize columns
#                     gram = W_norm.T @ W_norm  # Gram matrix
#                     identity = torch.eye(gram.size(0), device=W.device)
#                     ortho_loss += torch.norm(gram - identity, p='fro')**2
#         return ortho_loss
    
    
# %%
