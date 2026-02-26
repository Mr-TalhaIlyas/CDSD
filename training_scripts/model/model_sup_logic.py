#%%
import os
import math
import torch
import torch.nn as nn
from types import SimpleNamespace

from hgcn.hypergcn_large import Model
from model.crl.components import ConceptLogicLayers

class DivergenceLoss(nn.Module):
    def __init__(self):
        super(DivergenceLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        V, C = x[0].size()
        loss = 0

        for i in x:
            norm = torch.norm(i, dim=-1, keepdim=True, p=2)
            norm = norm @ norm.T
            loss_i = i @ i.T
            loss_i = loss_i / (norm + 1e-8)
            loss_p = self.relu(loss_i)
            loss_p = (loss_p.sum() - V) / (V * (V - 1))
            loss += loss_p

        return loss / len(x)
    
class GCN_Encoder(nn.Module):
    """
    Wrapper around pretrained HyperGCN Model.
    Loads pretrained weights from NTU-RGBD 120 classes.
    """
    def __init__(self, model_args):
        super(GCN_Encoder, self).__init__()
        # Make a copy to avoid modifying the original config
        model_args = dict(model_args)
        
        class_changed = False
        self.checkpoint_path = model_args.pop('pretrain_chkpt')
        
        if model_args['num_class'] != 120:
            print(f"!!! Warning !!!: Overriding num_class from {model_args['num_class']} to 120 for loading pretrained model.")
            orig_class = model_args['num_class']
            model_args['num_class'] = 120
            class_changed = True
        
        # Load pretrained model
        self.skeleton_model = Model(**model_args)
        
        # Load checkpoint
        state_dict = torch.load(self.checkpoint_path, map_location='cpu')
        self.skeleton_model.load_state_dict(state_dict)
        
        # put back original num_class
        if class_changed:
            model_args['num_class'] = orig_class
            
        print(30*'-')
        print(f" Loaded Hyper Graph GCN encoder from: {self.checkpoint_path}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print(30*'-')
        
    def forward(self, x):
        """
        Args:
            x: Input skeleton data (N, C, T, V, M)
        
        Returns:
            feature_concept: (N, 512) feature tensor
            hyper_feats: List of intermediate features
        """
        _, feature_concept, hyper_feats = self.skeleton_model(x)
        return feature_concept, hyper_feats 

#%%
class SkeletonACL_CLIP_Logic(nn.Module):
    """
    Skeleton Action Concept Learning (ACL) Model.
    
    Integrates:
    - Skeleton encoder (CTR-GCN)
    - CLIP text encoder (LoRA-enabled)
    - Concept predictors (per body part)
    - Concept reasoning layers (CRL logic)
    
    Logic layers are activated only after epoch 50 for staged training.
    """
    def __init__(self, cfg, device):
        super(SkeletonACL_CLIP_Logic, self).__init__()
        self.args = SimpleNamespace(**cfg)
        self.device = device
        
        self.gcn_dim = 512
        self.num_classes = self.args.model_args['num_class']
        self.total_concepts = self.args.model_args['spatial_concepts'] + self.args.model_args['temporal_concepts']

        # Make a copy of model_args to avoid modifying the original
        model_args = dict(self.args.model_args)
        self.logic_args = SimpleNamespace(**self.args.logic_model)
        
        # Remove concept keys if they exist (not needed)
        model_args.pop('spatial_concepts', None)
        model_args.pop('temporal_concepts', None)
        
        self.skeleton_model = GCN_Encoder(model_args=model_args).to(device)
        
        self.logic_layers = ConceptLogicLayers(
            n_concepts=self.total_concepts,
            n_actions=self.num_classes,  # 120 for NTU-120
            l1_dim=self.logic_args.l1_dim,
            l2_dim=self.logic_args.l2_dim,
            use_not=self.logic_args.use_not,
            use_skip=self.logic_args.use_skip,
            temperature=self.logic_args.temperature,
            l2_weight=self.logic_args.l2_weight
        ).to(device)
        
        self.fc = nn.Linear(self.gcn_dim, self.total_concepts).to(device)
        
        self.h_loss = DivergenceLoss().to(device)
        
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / self.num_classes))
        
        
        # weights = torch.tensor([0.4319960054164063, 0.06919220645937832, 4.899237259854465,
        #                         0.06369101582291781, 0.16093324129198336, 2.722887215160762,
        #                         0.17427728204012047, 0.38802758419180006, 0.08975818976216536])
        # weight=weights.to(device),
        self.loss_action = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        self.loss_concept = nn.BCEWithLogitsLoss().to(device)
  
        print("Initialized Simple Action Classification Model")
        print(f"  Action classes: {self.num_classes}")

        
    def forward(self, batch_data, batch_label, concepts_gt, epoch=None):
        # Get features from GCN encoder
        feats_concept, hyper_feats = self.skeleton_model(batch_data.to(self.device))
       
        # Predict concepts
        concept_logits = self.fc(feats_concept)
        # Predict actions
        action_logits = self.logic_layers(concept_logits)
        action_probs = torch.softmax(action_logits, dim=1)
        concept_probs = torch.sigmoid(concept_logits)
        # Compute loss
        loss_action = self.loss_action(action_logits, batch_label.to(self.device))
        loss_concept = self.loss_concept(concept_logits, concepts_gt)
        hyper_loss = self.h_loss(hyper_feats)
        
        loss = loss_action + hyper_loss + loss_concept
        
        if self.training:
            return action_probs, concept_probs, loss
        else:
            return action_probs, concept_probs