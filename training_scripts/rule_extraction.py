#%%
"""
Skeleton Visualization with Concept Activation and Rule Extraction
- Clean 2D skeleton drawing with body-part colored joints
- Joint size scaled by concept importance
- Prints activated concepts with their values
- Prints learned logical rules for the action
- Saves complete visualization with text panel as high-quality image

Adapted for VSVIG seizure detection pipeline:
- Uses SkeletonACL_CLIP_Logic model (model_sup_logic)
- Uses VSVIG feeder (feeder_vsvig)
- 9 action classes, 95 concepts from szr_cbm.csv
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(__file__))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#%%
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from pathlib import Path
import random
from collections import defaultdict
import textwrap
import math
from typing import Dict, List, Tuple, Sequence, Optional
from tqdm import tqdm

from feeders.feeder_vsvig import Feeder, custom_collate_fn, LABEL_NAMES, LABEL_MAP, SEIZURE_LABEL
from model.model_sup_logic import SkeletonACL_CLIP_Logic

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# =============================================================================
# BODY PART COLORS
# =============================================================================
BODY_PART_COLORS = {
    'head': '#E74C3C',
    'hand': '#3498DB',
    'arm': '#2ECC71',
    'hip': '#9B59B6',
    'leg': '#F39C12',
    'foot': '#1ABC9C',
    'temporal': '#607D8B',
    'interaction': '#795548',
}

BODY_PART_COLORS_LIGHT = {
    'head': '#F5A9A9',
    'hand': '#A9D0F5',
    'arm': '#A9F5A9',
    'hip': '#D0A9F5',
    'leg': '#F5D0A9',
    'foot': '#A9F5E1',
    'temporal': '#B8C6CF',
    'interaction': '#BCAAA4',
}

# Action class colors (seizure-focused)
ACTION_COLORS = {
    'sleeping': '#70B28C',
    'resting_or_lying_down': '#A6DCD3',
    'reading': '#8ba0a4',
    'play_with_phone_tablet': '#89A8D9',
    'eat_meal': '#C5DCE6',
    'talking': '#F2D6A2',
    'sitting_up': '#FFEDB8',
    'adjusting_position': '#BFA6A7',
    'seizure': '#F69EA7',
}

# =============================================================================
# JOINT MAPPING (25 NTU RGB+D joints)
# =============================================================================
JOINT_TO_BODY_PART = {
    0: 'hip', 1: 'hip', 2: 'hip', 3: 'head',
    4: 'arm', 5: 'arm', 6: 'arm', 7: 'hand',
    8: 'arm', 9: 'arm', 10: 'arm', 11: 'hand',
    12: 'leg', 13: 'leg', 14: 'leg', 15: 'foot',
    16: 'leg', 17: 'leg', 18: 'leg', 19: 'foot',
    20: 'hip', 21: 'hand', 22: 'hand', 23: 'hand', 24: 'hand',
}

NTU_SKELETON_CONNECTIONS = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]


def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# =============================================================================
# CONCEPT UTILITIES
# =============================================================================
def build_concept_to_body_part_mapping(concept_names):
    """Map concept indices to body parts based on concept name prefixes."""
    mapping = {}
    for i, name in enumerate(concept_names):
        name_lower = name.lower()
        if name_lower.startswith('head_'):
            mapping[i] = 'head'
        elif name_lower.startswith('hand_'):
            mapping[i] = 'hand'
        elif name_lower.startswith('arm_'):
            mapping[i] = 'arm'
        elif name_lower.startswith('hip_') or name_lower.startswith('torso_'):
            mapping[i] = 'hip'
        elif name_lower.startswith('leg_'):
            mapping[i] = 'leg'
        elif name_lower.startswith('foot_'):
            mapping[i] = 'foot'
        elif name_lower.startswith('interaction_'):
            mapping[i] = 'interaction'
        else:
            mapping[i] = 'temporal'
    return mapping


def compute_joint_importance(concept_probs, concept_to_body_part, threshold=0.5):
    """Compute importance score for each joint based on activated concepts."""
    body_part_importance = defaultdict(float)

    for c_idx, prob in enumerate(concept_probs):
        if prob > threshold:
            body_part = concept_to_body_part.get(c_idx, 'temporal')
            body_part_importance[body_part] += prob

    max_importance = max(body_part_importance.values()) if body_part_importance else 1.0
    if max_importance > 0:
        for part in body_part_importance:
            body_part_importance[part] /= max_importance

    joint_importance = {}
    for j_idx in range(25):
        body_part = JOINT_TO_BODY_PART.get(j_idx, 'hip')
        joint_importance[j_idx] = body_part_importance.get(body_part, 0.0)

    return joint_importance, dict(body_part_importance)


def get_activated_concepts(concept_probs, concept_names, concept_to_body_part, threshold=0.5):
    """Get list of activated concepts grouped by body part."""
    activated = defaultdict(list)

    for c_idx, prob in enumerate(concept_probs):
        if prob > threshold:
            body_part = concept_to_body_part.get(c_idx, 'temporal')
            activated[body_part].append((concept_names[c_idx], prob))

    for part in activated:
        activated[part] = sorted(activated[part], key=lambda x: -x[1])

    return dict(activated)


def get_body_part_indices(concept_names):
    """Group concept indices by body part."""
    body_parts = defaultdict(list)
    for i, name in enumerate(concept_names):
        name_lower = name.lower()
        for part in ['head', 'hand', 'arm', 'hip', 'leg', 'foot', 'interaction', 'temporal']:
            if name_lower.startswith(f'{part}_'):
                body_parts[part].append(i)
                break
        else:
            if 'temporal' in name_lower or 'dynamics' in name_lower or 'sequence' in name_lower:
                body_parts['temporal'].append(i)
    return dict(body_parts)


# =============================================================================
# SKELETON DRAWING
# =============================================================================
def draw_skeleton_with_concepts(
    ax, x, y, joint_importance,
    edge_color="black", linewidth=2.0,
    base_node_size=80.0, max_node_size=450.0,
    alpha=1.0, rank_gamma=1.2,
    missing_importance=0.5,
    use_lighter_colors=False,
):
    """
    Draw skeleton with joints colored by body part and sized by RANK of importance.
    Uses standard NTU RGB+D skeleton structure (25 joints).
    """
    colors = BODY_PART_COLORS_LIGHT if use_lighter_colors else BODY_PART_COLORS

    def clamp01(v):
        return max(0.0, min(1.0, v))

    joint_ids = list(range(25))
    imps = {j: float(joint_importance.get(j, missing_importance)) for j in joint_ids}
    sorted_items = sorted(imps.items(), key=lambda kv: (kv[1], kv[0]))

    n = len(sorted_items)
    if n <= 1:
        rank_map = {sorted_items[0][0]: 1.0} if n == 1 else {}
    else:
        rank_map = {j: i / (n - 1) for i, (j, _) in enumerate(sorted_items)}

    def size_from_rank(j):
        r = clamp01(rank_map.get(j, 0.0))
        r_scaled = r ** rank_gamma
        return base_node_size + r_scaled * (max_node_size - base_node_size), r_scaled

    zorder_edge = 1 if use_lighter_colors else 2
    for start, end in NTU_SKELETON_CONNECTIONS:
        ax.plot([x[start], x[end]], [y[start], y[end]],
                color=edge_color, linewidth=linewidth, alpha=alpha,
                solid_capstyle="round", zorder=zorder_edge)

    zorder_joint = 3 if use_lighter_colors else 5
    for j_idx in joint_ids:
        body_part = JOINT_TO_BODY_PART.get(j_idx, "hip")
        color = colors.get(body_part, "gray")
        node_size, r_scaled = size_from_rank(j_idx)
        lw = 0.6 + 2.0 * r_scaled
        ax.scatter(x[j_idx], y[j_idx], c=color, s=node_size,
                   zorder=zorder_joint, linewidths=lw, alpha=alpha)


# =============================================================================
# RULE EXTRACTOR (for SkeletonACL_CLIP_Logic / ConceptLogicLayers)
# =============================================================================
class RuleExtractor:
    """Extract and analyze learned logical rules from ConceptLogicLayers."""

    def __init__(self, model, concept_names, action_names):
        self.model = model
        self.concept_names = concept_names
        self.action_names = action_names
        self.logic_layers = model.logic_layers
        self.n_concepts = len(concept_names)
        self.n_actions = len(action_names)
        self.rules = None
        self.layer_info = None

    def extract_layer_info(self):
        """Extract information about each layer in the logic network."""
        layers = self.logic_layers.layer_list
        layer_info = []
        for i, layer in enumerate(layers):
            info = {
                'index': i,
                'type': type(layer).__name__,
                'input_dim': getattr(layer, 'input_dim', None),
                'output_dim': getattr(layer, 'output_dim', None),
            }
            if hasattr(layer, 'con_layer') and hasattr(layer, 'dis_layer'):
                info['con_W'] = layer.con_layer.W.detach().cpu().numpy()
                info['dis_W'] = layer.dis_layer.W.detach().cpu().numpy()
                info['con_dim'] = layer.con_layer.W.shape[1]
                info['dis_dim'] = layer.dis_layer.W.shape[1]
            if hasattr(layer, 'fc1'):
                info['weight'] = layer.fc1.weight.detach().cpu().numpy()
                info['bias'] = layer.fc1.bias.detach().cpu().numpy()
            layer_info.append(info)
        self.layer_info = layer_info
        return layer_info

    def extract_rules(self, top_k=10, weight_threshold=0.05):
        """Extract interpretable rules for each action class."""
        if self.layer_info is None:
            self.extract_layer_info()

        rules_per_action = {}
        lr_layer = self.layer_info[-1]
        W = lr_layer['weight']
        b = lr_layer['bias']

        for action_idx in range(min(self.n_actions, len(self.action_names))):
            action_name = self.action_names[action_idx]
            action_weights = W[action_idx]
            top_indices = np.argsort(np.abs(action_weights))[::-1][:top_k]

            rules = []
            for rank, hidden_idx in enumerate(top_indices):
                weight = action_weights[hidden_idx]
                if abs(weight) < weight_threshold:
                    continue
                contributing = self._trace_to_concepts(hidden_idx)
                rules.append({
                    'rank': rank + 1,
                    'weight': float(weight),
                    'hidden_idx': int(hidden_idx),
                    'polarity': 'positive' if weight > 0 else 'negative',
                    **contributing
                })
            rules_per_action[action_name] = rules

        self.rules = rules_per_action
        return rules_per_action

    def _trace_to_concepts(self, hidden_idx, threshold=0.3):
        """Trace hidden unit back to input concepts."""
        union_layer = None
        for layer in reversed(self.layer_info[:-1]):
            if 'con_W' in layer:
                union_layer = layer
                break

        if union_layer is None:
            return {'operator': 'UNKNOWN', 'concepts': []}

        con_dim = union_layer['con_dim']
        dis_dim = union_layer['dis_dim']
        total_dim = con_dim + dis_dim

        if hidden_idx < con_dim:
            W = union_layer['con_W']
            local_idx = hidden_idx
            op = 'AND'
        elif hidden_idx < total_dim:
            W = union_layer['dis_W']
            local_idx = hidden_idx - con_dim
            op = 'OR'
        else:
            skip_idx = hidden_idx - total_dim
            use_not = self.logic_layers.use_not
            n = self.n_concepts
            if use_not and skip_idx >= n:
                concept_idx = skip_idx - n
                if concept_idx < len(self.concept_names):
                    return {'operator': 'NOT', 'concepts': [('NOT', self.concept_names[concept_idx], 1.0)], 'is_skip': True}
            elif skip_idx < n and skip_idx < len(self.concept_names):
                return {'operator': 'DIRECT', 'concepts': [('', self.concept_names[skip_idx], 1.0)], 'is_skip': True}
            return {'operator': 'SKIP', 'concepts': [], 'is_skip': True}

        if local_idx >= W.shape[1]:
            return {'operator': op, 'concepts': [], 'note': 'idx out of range'}

        weights = W[:, local_idx]
        concepts = self._weights_to_concept_list(weights, threshold)
        return {'operator': op, 'concepts': concepts, 'is_skip': False}

    def _weights_to_concept_list(self, weights, threshold):
        """Convert weight vector to list of (negation, concept_name, weight)."""
        concepts = []
        n = self.n_concepts
        use_not = self.logic_layers.use_not
        input_dim = len(weights)

        for i, w in enumerate(weights):
            if w > threshold:
                if use_not and i >= input_dim // 2:
                    concept_idx = i - input_dim // 2
                    if concept_idx < n and concept_idx < len(self.concept_names):
                        concepts.append(('NOT', self.concept_names[concept_idx], float(w)))
                else:
                    if i < n and i < len(self.concept_names):
                        concepts.append(('', self.concept_names[i], float(w)))

        concepts.sort(key=lambda x: x[2], reverse=True)
        return concepts[:10]

    def format_rule_string(self, rule, max_concepts=5):
        """Format a rule as a human-readable string."""
        concepts = rule.get('concepts', [])
        op = rule.get('operator', 'UNKNOWN')

        if not concepts:
            if rule.get('is_skip'):
                return f"[skip: h{rule.get('hidden_idx', '?')}]"
            return "[no concepts]"

        concept_strs = []
        for neg, name, w in concepts[:max_concepts]:
            prefix = '\u00ac' if neg == 'NOT' else ''
            display_name = name.replace('_', ' ').title()
            if len(display_name) > 30:
                display_name = display_name[:27] + '...'
            concept_strs.append(f"{prefix}{display_name}")

        if op == 'AND':
            return ' \u2227 '.join(concept_strs)
        elif op == 'OR':
            return ' \u2228 '.join(concept_strs)
        elif op in ['DIRECT', 'NOT']:
            return concept_strs[0] if concept_strs else '[direct]'
        else:
            return ', '.join(concept_strs)

    def format_rules_for_display(self, action_name, top_n=5, sort_by_length=False,
                                   use_ascii_symbols=False, prefer_multi_concept=False,
                                   flip_weights=False, dashboard_top10=False):
        """Format rules for a specific action in human-readable format.

        Args:
            action_name: name of the action class
            top_n: max number of rules to return
            sort_by_length: if True, sort rules by number of concepts (longest first)
            use_ascii_symbols: if True, use AND/OR/NOT instead of unicode symbols
            prefer_multi_concept: if True, show multi-concept (AND/OR) rules first,
                then fill remaining slots with single-concept rules
            flip_weights: if True, negate rule weights for display (- -> +, + -> -)
            dashboard_top10: if True, use dashboard ordering:
                top 5 positive-weight (after flip) rules, next 3 multi-concept,
                remaining 2 from rest; skip connection rules are discarded
        """
        if self.rules is None:
            self.extract_rules()

        if action_name not in self.rules:
            return [f"No rules found for: {action_name}"]

        # Use all available rules (not just top_n) so we have a larger pool
        all_rules = self.rules[action_name]
        formatted = []

        for rule in all_rules:
            concepts = rule.get('concepts', [])
            op = rule.get('operator', 'UNKNOWN')
            is_skip = rule.get('is_skip', False)

            # In dashboard mode, discard only empty skip connections (no concepts).
            # Keep DIRECT and NOT skip connections — they are valid single-concept rules.
            if dashboard_top10 and is_skip and not concepts:
                continue

            if not concepts:
                if is_skip:
                    formatted.append((0, False, 0.0, f"r{rule['rank']}: [skip connection] (w: {rule['weight']:+.3f})"))
                continue

            concept_strs = []
            for neg, name, w in concepts[:5]:
                if use_ascii_symbols:
                    prefix = 'NOT ' if neg == 'NOT' else ''
                else:
                    prefix = '\u00ac' if neg == 'NOT' else ''
                concept_strs.append(f"{prefix}{name}")

            if concept_strs:
                n_concepts = len(concept_strs)
                is_multi = (n_concepts >= 2 and op in ('AND', 'OR'))

                if op in ('DIRECT', 'NOT'):
                    rule_str = concept_strs[0]
                else:
                    if use_ascii_symbols:
                        op_str = ' AND ' if op == 'AND' else ' OR '
                    else:
                        op_str = ' \u2227 ' if op == 'AND' else ' \u2228 '
                    rule_str = op_str.join(concept_strs)

                display_weight = -rule['weight'] if flip_weights else rule['weight']
                formatted.append((n_concepts, is_multi, display_weight,
                                  f"r{rule['rank']}: {rule_str}  (w: {display_weight:+.3f})"))

        if dashboard_top10:
            # Dashboard ordering: 5 positive + 3 multi-concept + 2 remaining
            # Sort all by display_weight descending (highest = most supporting)
            all_sorted = sorted(formatted, key=lambda x: x[2], reverse=True)
            positive = [x for x in all_sorted if x[2] > 0]

            # Pick up to 5 positive-weight rules
            selected = positive[:5]
            selected_texts = {t for _, _, _, t in selected}
            # If fewer than 5 positive, fill from top by weight (regardless of sign)
            for item in all_sorted:
                if item[3] not in selected_texts and len(selected) < 5:
                    selected.append(item)
                    selected_texts.add(item[3])

            # Pick next 3 multi-concept (not already selected)
            multi = [x for x in all_sorted if x[1]]
            for item in multi:
                if item[3] not in selected_texts and len(selected) < 8:
                    selected.append(item)
                    selected_texts.add(item[3])
            # Fill remaining from all formatted (not already selected)
            for item in all_sorted:
                if item[3] not in selected_texts and len(selected) < 10:
                    selected.append(item)
                    selected_texts.add(item[3])

            result = [text for _, _, _, text in selected[:top_n]]
        elif prefer_multi_concept:
            multi = [(nc, m, dw, t) for nc, m, dw, t in formatted if m]
            single = [(nc, m, dw, t) for nc, m, dw, t in formatted if not m]
            multi.sort(key=lambda x: x[0], reverse=True)
            formatted = multi + single
            result = [text for _, _, _, text in formatted[:top_n]]
        elif sort_by_length:
            formatted.sort(key=lambda x: x[0], reverse=True)
            result = [text for _, _, _, text in formatted[:top_n]]
        else:
            result = [text for _, _, _, text in formatted[:top_n]]

        return result if result else [f"No significant rules for: {action_name}"]

    def get_concept_usage_matrix(self):
        """Compute concept usage matrix: [n_actions, n_concepts]."""
        if self.rules is None:
            self.extract_rules()

        usage = np.zeros((self.n_actions, self.n_concepts))
        for action_idx, action_name in enumerate(self.action_names):
            if action_name not in self.rules:
                continue
            for rule in self.rules[action_name]:
                weight = abs(rule['weight'])
                for neg, concept_name, concept_weight in rule.get('concepts', []):
                    try:
                        concept_idx = self.concept_names.index(concept_name)
                        sign = -1 if neg == 'NOT' else 1
                        usage[action_idx, concept_idx] += sign * weight * concept_weight
                    except ValueError:
                        continue
        return usage


# =============================================================================
# TEXT PANEL DRAWING
# =============================================================================
def draw_text_panel(ax, activated_concepts, formatted_rules, predicted_action,
                    gt_action=None, is_correct=True):
    """Draw activated concepts and learned rules as text on an axis."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    y_pos = 0.98
    line_height = 0.08

    if is_correct:
        ax.text(0.5, y_pos, f"Predicted: {predicted_action.replace('_', ' ').title()}",
               fontsize=12, fontweight='bold', ha='center', va='top',
               transform=ax.transAxes, color='#2E7D32')
    else:
        ax.text(0.5, y_pos, f"Predicted: {predicted_action.replace('_', ' ').title()}",
               fontsize=12, fontweight='bold', ha='center', va='top',
               transform=ax.transAxes, color='#C62828')
        y_pos -= line_height
        ax.text(0.5, y_pos, f"Ground Truth: {gt_action.replace('_', ' ').title()}",
               fontsize=12, fontweight='bold', ha='center', va='top',
               transform=ax.transAxes, color='#2E7D32')

    y_pos -= line_height * 1.2
    ax.axhline(y=y_pos, xmin=0.02, xmax=0.98, color='gray', linewidth=0.5)
    y_pos -= line_height * 0.5

    # ACTIVATED CONCEPTS
    ax.text(0.02, y_pos, "ACTIVATED CONCEPTS", fontsize=11, fontweight='bold',
           va='top', transform=ax.transAxes, color='#333333')
    y_pos -= line_height * 0.9

    body_parts_order = ['head', 'hand', 'arm', 'hip', 'leg', 'foot', 'temporal', 'interaction']
    for part in body_parts_order:
        if part in activated_concepts and activated_concepts[part]:
            color = BODY_PART_COLORS.get(part, '#666666')
            ax.text(0.03, y_pos, f"\u25cf {part.upper()}:", fontsize=9, fontweight='bold',
                   va='top', transform=ax.transAxes, color=color)
            concepts_str = "  ".join([f"{name}: {prob:.3f}" for name, prob in activated_concepts[part][:4]])
            ax.text(0.14, y_pos, concepts_str, fontsize=8, va='top',
                   transform=ax.transAxes, color='#444444')
            y_pos -= line_height * 0.8

    y_pos -= line_height * 0.3
    ax.axhline(y=y_pos, xmin=0.02, xmax=0.98, color='gray', linewidth=0.5)
    y_pos -= line_height * 0.5

    # LEARNED RULES
    ax.text(0.02, y_pos, "LEARNED RULES", fontsize=11, fontweight='bold',
           va='top', transform=ax.transAxes, color='#333333')
    y_pos -= line_height * 0.9

    if isinstance(formatted_rules, list) and formatted_rules:
        for rule_str in formatted_rules[:5]:
            ax.text(0.03, y_pos, rule_str, fontsize=9, va='top',
                   transform=ax.transAxes, color='#444444', family='monospace')
            y_pos -= line_height * 0.8
    else:
        ax.text(0.03, y_pos, "No rules extracted", fontsize=9, va='top',
               transform=ax.transAxes, color='#888888', style='italic')


# =============================================================================
# MAIN VISUALIZATION FUNCTION
# =============================================================================
def visualize_skeleton_with_concepts(pose_data, concept_probs, concept_names,
                                     concept_to_body_part, predicted_action,
                                     gt_action=None, predicted_label=None, gt_label=None,
                                     rule_extractor=None,
                                     person_idx=0, n_frames=8,
                                     save_path=None, threshold=0.5, dpi=300):
    """
    Visualize skeleton frames with concept-based coloring and sizing.
    Includes text panel with activated concepts and learned rules.
    """
    C, T, V, M = pose_data.shape

    is_correct = True
    if gt_label is not None and predicted_label is not None:
        is_correct = (gt_label == predicted_label)
    elif gt_action is not None:
        is_correct = (predicted_action == gt_action)

    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)

    joint_importance, body_part_importance = compute_joint_importance(
        concept_probs, concept_to_body_part, threshold
    )
    activated_concepts = get_activated_concepts(
        concept_probs, concept_names, concept_to_body_part, threshold
    )

    formatted_rules = []
    if rule_extractor is not None:
        formatted_rules = rule_extractor.format_rules_for_display(predicted_action, top_n=5)

    # Print to console
    print("\n" + "=" * 70)
    if is_correct:
        print(f"\u2713 CORRECT - Predicted: {predicted_action.upper().replace('_', ' ')}")
    else:
        print(f"\u2717 WRONG - Predicted: {predicted_action.upper().replace('_', ' ')}")
        print(f"  Ground Truth: {gt_action.upper().replace('_', ' ')}")
    print("=" * 70)

    print("\nACTIVATED CONCEPTS:")
    print("-" * 50)
    for part in ['head', 'hand', 'arm', 'hip', 'leg', 'foot', 'temporal', 'interaction']:
        if part in activated_concepts and activated_concepts[part]:
            print(f"\n  {part.upper()}:")
            for concept_name, prob in activated_concepts[part]:
                print(f"    \u2022 {concept_name}: {prob:.3f}")

    if rule_extractor is not None:
        print("\n" + "-" * 50)
        print("LEARNED RULES:")
        print("-" * 50)
        if isinstance(formatted_rules, list):
            for rule_str in formatted_rules:
                print(f"  {rule_str}")
    print("-" * 50)

    # Create figure
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 1, height_ratios=[2, 1.2], hspace=0.12)

    gs_skeletons = gs[0].subgridspec(1, n_frames, wspace=0.05)
    skeleton_axes = [fig.add_subplot(gs_skeletons[0, i]) for i in range(n_frames)]

    for i, frame_idx in enumerate(frame_indices):
        ax = skeleton_axes[i]

        x1 = pose_data[0, frame_idx, :, person_idx]
        y1 = pose_data[1, frame_idx, :, person_idx]

        has_second_person = False
        if M >= 2:
            second_person_idx = 1 if person_idx == 0 else 0
            x2 = pose_data[0, frame_idx, :, second_person_idx]
            y2 = pose_data[1, frame_idx, :, second_person_idx]
            if np.abs(x2).sum() > 0.01 or np.abs(y2).sum() > 0.01:
                has_second_person = True

        if np.abs(x1).sum() < 0.01 and np.abs(y1).sum() < 0.01:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        else:
            if has_second_person:
                draw_skeleton_with_concepts(
                    ax, x2, y2, joint_importance,
                    edge_color='#AAAAAA', linewidth=1.0,
                    base_node_size=15, max_node_size=60,
                    alpha=0.4, rank_gamma=1.2,
                    missing_importance=0.5,
                    use_lighter_colors=True
                )

            draw_skeleton_with_concepts(
                ax, x1, y1, joint_importance,
                edge_color='black', linewidth=1.5,
                base_node_size=25, max_node_size=120,
                alpha=1.0, use_lighter_colors=False
            )

            all_x = [x1]
            all_y = [y1]
            if has_second_person:
                all_x.append(x2)
                all_y.append(y2)
            all_x = np.concatenate(all_x)
            all_y = np.concatenate(all_y)

            valid_mask = (np.abs(all_x) > 0.001) | (np.abs(all_y) > 0.001)
            if valid_mask.sum() > 0:
                valid_x = all_x[valid_mask]
                valid_y = all_y[valid_mask]
                x_margin = (valid_x.max() - valid_x.min()) * 0.15 + 0.05
                y_margin = (valid_y.max() - valid_y.min()) * 0.15 + 0.05
                ax.set_xlim(valid_x.min() - x_margin, valid_x.max() + x_margin)
                ax.set_ylim(valid_y.min() - y_margin, valid_y.max() + y_margin)
            else:
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Frame {frame_idx + 1}', fontsize=10, pad=5)

    # Bottom: Text panel
    ax_text = fig.add_subplot(gs[1])
    draw_text_panel(ax_text, activated_concepts, formatted_rules,
                   predicted_action, gt_action, is_correct)

    title_color = '#2E7D32' if is_correct else '#C62828'
    status = "\u2713" if is_correct else "\u2717"
    fig.suptitle(f"{status} Action: {predicted_action.replace('_', ' ').title()}",
                fontsize=16, fontweight='bold', y=0.98, color=title_color)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi,
                   facecolor='white', edgecolor='none')

    plt.close(fig)
    return fig


# =============================================================================
# MODEL LOADING & INFERENCE
# =============================================================================
def load_model_and_get_concepts(config_path, checkpoint_path, concept_csv_path, device='cuda'):
    """
    Load the SkeletonACL_CLIP_Logic model and concept information.

    Returns:
        model, args, concept_names, action_names, concept_to_body_part, rule_extractor
    """
    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)

    model = SkeletonACL_CLIP_Logic(args, device).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Model loaded from: {checkpoint_path}")

    concept_df = pd.read_csv(concept_csv_path)
    concept_names = [c for c in concept_df.columns.tolist() if c != 'action_class']
    concept_to_body_part = build_concept_to_body_part_mapping(concept_names)

    action_names = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    print(f"Loaded {len(concept_names)} concepts, {len(action_names)} action classes")

    rule_extractor = RuleExtractor(model, concept_names, action_names)
    rule_extractor.extract_rules(top_k=10, weight_threshold=0.05)

    return model, args, concept_names, action_names, concept_to_body_part, rule_extractor


@torch.no_grad()
def get_concept_probs_and_prediction(model, batch_data, device):
    """
    Get concept probabilities and model prediction for a batch.

    Directly calls model subcomponents to avoid requiring concepts_gt
    in the forward pass (which computes loss even in eval mode).

    Returns:
        concept_probs: (B, n_concepts) in [0, 1]
        predicted_labels: (B,) integer class indices
        action_probs: (B, n_classes) softmax probabilities
    """
    batch_data = batch_data.to(device)

    feats_concept, hyper_feats = model.skeleton_model(batch_data)
    concept_logits = model.fc(feats_concept)
    action_logits = model.logic_layers(concept_logits)

    action_probs = torch.softmax(action_logits, dim=1).cpu().numpy()
    concept_probs = torch.sigmoid(concept_logits).cpu().numpy()
    predicted_labels = action_probs.argmax(axis=1)

    return concept_probs, predicted_labels, action_probs


# =============================================================================
# BATCH VISUALIZATION (save all test samples)
# =============================================================================
def run_batch_visualization(
    config_path, checkpoint_path, concept_csv_path,
    output_dir='./skeleton_concepts',
    max_samples=None,
    n_frames=8,
    threshold=0.5,
    save_correct=True,
    save_wrong=True,
    filter_class=None,
):
    """
    Run skeleton + concept + rule visualization on test samples.

    Args:
        config_path: Path to VSVIG config YAML
        checkpoint_path: Path to model checkpoint
        concept_csv_path: Path to concept bank CSV
        output_dir: Directory to save output images
        max_samples: Maximum samples to visualize (None=all)
        n_frames: Number of skeleton frames per sample
        threshold: Concept activation threshold
        save_correct: Save correctly classified samples
        save_wrong: Save wrongly classified samples
        filter_class: Only visualize this class (e.g., 'seizure')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'correct').mkdir(exist_ok=True)
    (output_dir / 'wrong').mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("BATCH SKELETON + CONCEPT + RULE VISUALIZATION")
    print("=" * 60)

    model, args, concept_names, action_names, concept_to_body_part, rule_extractor = \
        load_model_and_get_concepts(config_path, checkpoint_path, concept_csv_path, device)

    init_seed(args.get('seed', 1))

    test_loader = torch.utils.data.DataLoader(
        dataset=Feeder(**args['test_feeder_args']),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    total_samples = len(test_loader.dataset)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)

    print(f"\nVisualizing up to {total_samples} samples...")

    n_correct = 0
    n_wrong = 0
    n_saved = 0

    for i, (batch_data, batch_label, batch_concept_vecs, batch_prompts) in enumerate(tqdm(test_loader, total=total_samples)):
        if max_samples is not None and i >= max_samples:
            break

        gt_label = batch_label[0].item()
        gt_action = action_names[gt_label] if gt_label < len(action_names) else f"Class_{gt_label}"

        if filter_class is not None and gt_action != filter_class:
            continue

        concept_probs, predicted_labels, action_probs = get_concept_probs_and_prediction(
            model, batch_data, device
        )
        predicted_label = predicted_labels[0]
        predicted_action = action_names[predicted_label] if predicted_label < len(action_names) else f"Class_{predicted_label}"

        is_correct = (gt_label == predicted_label)

        if is_correct:
            n_correct += 1
            if not save_correct:
                continue
            subdir = 'correct'
        else:
            n_wrong += 1
            if not save_wrong:
                continue
            subdir = 'wrong'

        status_str = "correct" if is_correct else "wrong"
        save_path = output_dir / subdir / f"sample_{i:05d}_{status_str}_gt-{gt_action}_pred-{predicted_action}.png"

        visualize_skeleton_with_concepts(
            batch_data[0].cpu().numpy(),
            concept_probs[0],
            concept_names, concept_to_body_part,
            predicted_action, gt_action=gt_action,
            predicted_label=predicted_label, gt_label=gt_label,
            rule_extractor=rule_extractor,
            person_idx=0, n_frames=n_frames,
            save_path=str(save_path), threshold=threshold,
            dpi=200
        )
        n_saved += 1

    total_processed = n_correct + n_wrong
    acc = n_correct / total_processed * 100 if total_processed > 0 else 0

    print("\n" + "=" * 60)
    print("BATCH VISUALIZATION COMPLETE")
    print(f"  Total processed: {total_processed}")
    print(f"  Correct: {n_correct} ({acc:.1f}%)")
    print(f"  Wrong: {n_wrong} ({100 - acc:.1f}%)")
    print(f"  Saved: {n_saved} images")
    print(f"  Output: {output_dir}")
    print("=" * 60)


# =============================================================================
# SINGLE SAMPLE VISUALIZATION
# =============================================================================
def run_single_sample(
    config_path, checkpoint_path, concept_csv_path,
    sample_idx=0,
    output_dir='./skeleton_concepts',
    n_frames=8,
    threshold=0.5,
):
    """Run visualization for a single test sample by index."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, args, concept_names, action_names, concept_to_body_part, rule_extractor = \
        load_model_and_get_concepts(config_path, checkpoint_path, concept_csv_path, device)

    init_seed(args.get('seed', 1))

    test_loader = torch.utils.data.DataLoader(
        dataset=Feeder(**args['test_feeder_args']),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    print(f"\nLooking for sample index {sample_idx}...")

    for i, (batch_data, batch_label, batch_concept_vecs, batch_prompts) in enumerate(test_loader):
        if i == sample_idx:
            gt_label = batch_label[0].item()
            gt_action = action_names[gt_label]

            concept_probs, predicted_labels, action_probs = get_concept_probs_and_prediction(
                model, batch_data, device
            )
            predicted_label = predicted_labels[0]
            predicted_action = action_names[predicted_label]

            is_correct = (gt_label == predicted_label)
            status_str = "correct" if is_correct else "wrong"
            save_path = output_dir / f"sample_{i}_{status_str}_{predicted_action}.png"

            fig = visualize_skeleton_with_concepts(
                batch_data[0].cpu().numpy(),
                concept_probs[0],
                concept_names, concept_to_body_part,
                predicted_action, gt_action=gt_action,
                predicted_label=predicted_label, gt_label=gt_label,
                rule_extractor=rule_extractor,
                person_idx=0, n_frames=n_frames,
                save_path=str(save_path), threshold=threshold
            )

            # Print action probabilities
            print("\nAction Probabilities:")
            sorted_indices = np.argsort(action_probs[0])[::-1]
            for idx in sorted_indices:
                name = action_names[idx] if idx < len(action_names) else f"Class_{idx}"
                marker = " << GT" if idx == gt_label else ""
                marker += " << PRED" if idx == predicted_label else ""
                print(f"  {name:30s}: {action_probs[0][idx]:.4f}{marker}")

            return fig

        if i > sample_idx:
            break

    print(f"Sample index {sample_idx} not found (dataset has {len(test_loader.dataset)} samples)")
    return None


# =============================================================================
# MAIN
# =============================================================================
def main():
    config_path = '/mnt/ssd/Talha/reason/config/szr/config_skeleton_vsvig.yaml'
    checkpoint_path = '/mnt/ssd/Talha/reason/work_dir/vsvig3_f1_logic2/best_model.pth'
    concept_csv_path = '/mnt/ssd/Talha/reason/concepts/szr_cbm.csv'
    output_dir = '/mnt/ssd/Talha/reason/skeleton_concepts'

    # ---- Mode 1: Visualize a single sample ----
    # run_single_sample(
    #     config_path, checkpoint_path, concept_csv_path,
    #     sample_idx=0,
    #     output_dir=output_dir,
    #     n_frames=8,
    #     threshold=0.5,
    # )

    # ---- Mode 2: Batch visualize test samples ----
    run_batch_visualization(
        config_path, checkpoint_path, concept_csv_path,
        output_dir=output_dir,
        max_samples=None,         # Set to None for all
        n_frames=8,
        threshold=0.5,
        save_correct=True,
        save_wrong=True,
        filter_class='seizure',      # Set to 'seizure' for seizure-only
    )


if __name__ == '__main__':
    main()

#%%
