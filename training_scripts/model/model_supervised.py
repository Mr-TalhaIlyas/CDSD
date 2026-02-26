#%%
import os
import math
import torch
import torch.nn as nn
from types import SimpleNamespace

from hgcn.hypergcn_large import Model

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
        
        # Make a copy of model_args to avoid modifying the original
        model_args = dict(self.args.model_args)
        
        # Remove concept keys if they exist (not needed)
        model_args.pop('spatial_concepts', None)
        model_args.pop('temporal_concepts', None)
        
        self.skeleton_model = GCN_Encoder(model_args=model_args).to(device)
        self.fc = nn.Linear(self.gcn_dim, self.num_classes).to(device)
        self.h_loss = DivergenceLoss().to(device)
        
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / self.num_classes))
        
        
        # weights = torch.tensor([0.4319960054164063, 0.06919220645937832, 4.899237259854465,
        #                         0.06369101582291781, 0.16093324129198336, 2.722887215160762,
        #                         0.17427728204012047, 0.38802758419180006, 0.08975818976216536])
        # weight=weights.to(device),
        self.loss_action = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

  
        print("Initialized Simple Action Classification Model")
        print(f"  Action classes: {self.num_classes}")

        
    def forward(self, batch_data, batch_label, batch_concept_vectors_by_part=None, batch_prompts_by_part=None,
                epoch=None, total_epochs=None, warmup_epochs=None):
        
        # Get features from GCN encoder
        feats_concept, hyper_feats = self.skeleton_model(batch_data)
       
        # Predict actions
        action_logits = self.fc(feats_concept)
        action_probs = torch.softmax(action_logits, dim=1)
        
        # Compute loss
        loss = self.loss_action(action_logits, batch_label.to(self.device))
        
        hyper_loss = self.h_loss(hyper_feats)
        
        loss += hyper_loss
        
        if self.training:
            return action_probs, loss
        else:
            return action_probs