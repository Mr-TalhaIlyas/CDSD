#%%
import os
# os.chdir(os.path.dirname(__file__))
# os.chdir('/data_hdd/talha/nsai/scal/')

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

# from model.ctrgcn import Model_lst_4part
import clip
from model.baseline import TextCLIP
from hgcn.hypergcn_large import Model


from loralib.utils import apply_lora, mark_only_lora_as_trainable

from model.multi_pos_clip import MultiPositiveClipLoss, MultiPositiveClipConfig
from model.crl.components import ConceptLogicLayers
from model.ml_decoder import MLDecoder

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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# with open('/data_hdd/talha/nsai/scal/config/config_skeleton_crl.yaml', 'r') as f:
#     args = yaml.safe_load(f)
    
# args = SimpleNamespace(**args)


# skeleton_model = Model_lst_4part(**args.model_args).to(device)

#%%

# chkpt = '/mnt/ssd/Talha/scal/runs-110-34650.pt'
# chkpt = '/data_hdd/talha/nsai/scal/hgcn/hgcn-runs-122-120048.pt'
chkpt_clip = '/mnt/ssd/Talha/scal/runs-text_encoder-110-34650.pt'

# skeleton_model.load_state_dict(torch.load(chkpt))
# %%
class GCN_Encoder(nn.Module):
    """
    Wrapper around pretrained Model_lst_4part that:
    1. Loads pretrained weights
    2. Freezes all parameters
    3. Only returns feature_dict (CLIP-aligned features)
    """
    def __init__(self, model_args):
        super(GCN_Encoder, self).__init__()
        class_changed = False
        self.checkpoint_path = model_args['pretrain_chkpt']
        if model_args['num_class'] != 120:
            print(f"!!! Warning !!!: Overriding num_class from {model_args['num_class']} to 120 for loading pretrained model.")
            orig_class = model_args['num_class']
            model_args['num_class'] = 120
            class_changed = True
        del model_args['pretrain_chkpt']
        # Load pretrained model
        self.skeleton_model = Model(**model_args)
        
        # Load checkpoint
        state_dict = torch.load(self.checkpoint_path, map_location='cpu')
        self.skeleton_model.load_state_dict(state_dict)
        # put back original num_class
        if class_changed:
            model_args['num_class'] = orig_class
        # freeze all parameters
        # for param in self.skeleton_model.parameters():
        #     param.requires_grad = False
        print(30*'-')
        print(f" Loaded Hyper Graph GCN encoder from: {self.checkpoint_path}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print(30*'-')
    def forward(self, x):
        """
        Args:
            x: Input skeleton data (N, C, T, V, M) or (N, T, V*C)
        
        Returns:
            feature_dict: Dictionary of CLIP-aligned features
                - Keys: model names (e.g., 'ViT-B/32')
                - Values: (N, 512) feature tensors
        """
        _, feature_concept, hyper_feats = self.skeleton_model(x)
        
        return feature_concept, hyper_feats 

class Text_Encoder(nn.Module):
    def __init__(self, lora_args, device, checkpoint_path=chkpt_clip):
        super(Text_Encoder, self).__init__()
        
        self.full_finetune = False
        self.lora_args = lora_args
        self.device = device
        clip_model, _ = clip.load(self.lora_args.backbone, device=device)
        
        if not self.full_finetune:
            # apply LoRA to CLIP model
            _ = apply_lora(self.lora_args, clip_model)
            mark_only_lora_as_trainable(clip_model)
        
        del clip_model.visual
        self.text_encoder = nn.ModuleDict()
        clip_model = TextCLIP(clip_model)
        
        self.text_encoder[self.lora_args.backbone] = clip_model # text encoder that accepts tokenized text input
        if not self.full_finetune:
            self.text_encoder.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.text_encoder.to(device)
        
        print(30*'-')
        if self.full_finetune:
            print(f"   • Fully fine-tuning CLIP text encoder: {self.lora_args.backbone}")
        print(f"   • CLIP backbone: {self.lora_args.backbone}")
        print(f"   • CLIP trainable params: {sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad) / 1e6:.2f}M")
        print(30*'-')
    def forward(self, all_prompts):
        """
        Memory-optimized CLIP text feature extraction with trainable support.
        Processes all body parts in a single forward pass for efficiency.
        """

        # Tokenize everything at once
        text_tokens = clip.tokenize(all_prompts, truncate=False).to(self.device)
        
        # Single forward pass for all body parts
        text_features = self.text_encoder[self.lora_args.backbone](text_tokens).float()

        return text_features
    
class ConceptRefiner(nn.Module):
    """
    Refines concept logits based on co-occurrence patterns.
    Works in logit space and keeps them as logits.
    """
    def __init__(self, n_concepts, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_concepts)
            # REMOVED Sigmoid: We want to output negative values for suppression
        )
        # Learnable scaling factor, initialized small for stability
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # x: [B, n_concepts] (Logits)
        delta = self.net(x)
        return x + self.alpha * delta
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
        self.lora_args = SimpleNamespace(**self.args.lora_args)
        self.logic_args = SimpleNamespace(**self.args.logic_model)
        
        self.use_concept_refiner = self.args.use_concept_refiner
        self.gcn_dim = 512
        self.clip_dim = 512
        self.concept_smoothing = self.args.concept_smoothing
        
        self.total_concepts = self.args.model_args['spatial_concepts'] + self.args.model_args['temporal_concepts']
        self.spatial_concepts = self.args.model_args['spatial_concepts']
        self.temporal_concepts = self.args.model_args['temporal_concepts']
        
        del self.args.model_args['spatial_concepts']
        del self.args.model_args['temporal_concepts']
        
        self.skeleton_model = GCN_Encoder(model_args=self.args.model_args).to(device)
        self.text_encoder = Text_Encoder(self.lora_args, device)
        
        self.text_projection = nn.Sequential(nn.Linear(512, 512),
                                             nn.LayerNorm(512),
                                             nn.ReLU(),
                                             nn.Linear(512, 512)).to(device)
        
        self.skel_projection = nn.Sequential(
                                            nn.Linear(self.gcn_dim, self.gcn_dim),
                                            nn.LayerNorm(512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512)
                                        )
        
        # self.fc = nn.Sequential(
        #                         nn.Linear(self.gcn_dim, self.gcn_dim),
        #                         nn.ReLU(),
        #                         nn.Dropout(0.05),
        #                         nn.Linear(self.gcn_dim, self.total_concepts)
        #                     ).to(device)
        self.fc = MLDecoder(
                num_classes=self.total_concepts,           # 78 concepts
                num_of_groups=64,      # 16         # 40 query embeddings
                decoder_embedding=512,       # 512 hidden dim
                initial_num_features=self.gcn_dim, # 256 (GCN channels) ← KEY!
                zsl=0  # Not using zero-shot learning
            ).to(device)
        
        if self.use_concept_refiner:
            print(" * Using Concept Refiner module.")
            self.concept_refiner = ConceptRefiner(
                n_concepts=self.total_concepts, # 78
                hidden_dim=256
            ).to(device)
        
        # Logic layers - always created but only used after epoch 50
        self.logic_layers = ConceptLogicLayers(
            n_concepts=self.total_concepts,
            n_actions=self.args.model_args['num_class'],  # 120 for NTU-120
            l1_dim=self.logic_args.l1_dim,
            l2_dim=self.logic_args.l2_dim,
            use_not=self.logic_args.use_not,
            use_skip=self.logic_args.use_skip,
            temperature=self.logic_args.temperature,
            l2_weight=self.logic_args.l2_weight
        ).to(device)

        mplclip_cfg = MultiPositiveClipConfig(
                        init_logit_scale=0.0,
                        symmetric=False,
                        use_ddp_all_gather=False,  # set True under DDP for stronger negatives
                        label_smoothing=self.concept_smoothing
                    )
                            
        self.loss_concept = nn.BCEWithLogitsLoss().cuda(device)
        # self.loss_concept = AsymmetricLossOptimized(gamma_pos=0.0, gamma_neg=4.0, clip=0.05).to(device)
        # self.loss_concept = TwoWayLoss(Tp=4., Tn=1.).to(device)
        self.mp_clip_loss = MultiPositiveClipLoss(mplclip_cfg).to(device)
        self.h_loss = DivergenceLoss().to(device)
        self.loss_action = nn.CrossEntropyLoss(label_smoothing=0.1).cuda(device)

        if self.concept_smoothing > 0.0:
            print(f" * Using concept label smoothing: {self.concept_smoothing}")
            # print(" * Using Extneded LOGIC layers with IMPLIES.")
        else:
            print(f" * No concept label smoothing.")
        print('Total concepts:', self.total_concepts)
        print('L2 weight for logic layers:', self.logic_args.l2_weight)
    def _parse_prompts(self, batch_prompts_by_part):
        all_prompts = batch_prompts_by_part['full_body'] + batch_prompts_by_part['temporal']
        # indices_to_remove = [50, 65] # 53

        # for i in sorted(indices_to_remove, reverse=True):
        #     all_prompts.pop(i)
        return all_prompts

    # def update_temperature(self, epoch, total_epochs):
    #     """Anneal temperature for logic layers from 1.0 down to 0.01"""
    #     # Linear decay or Cosine decay
    #     # t = 1.0 - (epoch / total_epochs) # Linear decay to 0
    #     # t = max(t, 0.01) # Clip at 0.01
        
    #     # Or delegate to the internal method if your ConceptLogicLayers has it:
    #     if hasattr(self.logic_layers, 'anneal_temperature'):
    #         self.logic_layers.anneal_temperature(epoch, total_epochs)
    #     else:
    #         # Fallback: Manual annealing if the layer doesn't have the method
    #         # Assuming logic layers have a .temperature attribute
    #         if hasattr(self.logic_layers, 'temperature'):
    #             new_temp = 1.0 - (0.99 * (epoch / total_epochs))
    #             self.logic_layers.temperature = new_temp
    
    def forward(self, batch_data, batch_label, batch_concept_vectors_by_part, batch_prompts_by_part,
                epoch=None, total_epochs=None, warmup_epochs=None):
        
        if self.training and (epoch is None or total_epochs is None or warmup_epochs is None):
            raise ValueError("Epoch, total_epochs, and warmup_epochs must be provided during training.")
        
        # Extract features from skeleton encoder
        feats_concept, hyper_feats = self.skeleton_model(batch_data)
        
        concept_logits = self.fc(feats_concept.unsqueeze(1))  # [B, 78]
        
        if self.use_concept_refiner:
            concept_logits = self.concept_refiner(concept_logits)
        
        action_logits = self.logic_layers(concept_logits)  # [B, 120]
        action_probs = torch.softmax(action_logits, dim=1)
        
        if self.training:
            # Parse text prompts and encode
            all_prompts = self._parse_prompts(batch_prompts_by_part)
            feats_txt = self.text_encoder(all_prompts)  # [C, 512]
        
            # Project features
            feats_skel = self.skel_projection(feats_concept)
            feats_txt = self.text_projection(feats_txt)
        
            # Prepare concept ground truth
            concepts_gt = torch.cat(
                (batch_concept_vectors_by_part['full_body'].float().to(self.device),
                 batch_concept_vectors_by_part['temporal'].float().to(self.device)),
                dim=1)  # [B, 74]
            pos_mask = concepts_gt.bool() # create positive mask for contrastive loss FIRST
            if self.concept_smoothing > 0.0:
                # create smoothed version of concepts_gt
                epsilon = self.concept_smoothing
                concepts_gt = concepts_gt * (1 - epsilon) + 0.5 * epsilon
            
            # Calculate concept loss (always active)
            loss_concept = self.loss_concept(concept_logits, concepts_gt)
            
            # Calculate action loss (always active)
            action_loss = self.loss_action(action_logits, batch_label.to(self.device))
            
            # Calculate contrastive loss (always active)
            
            loss_cont, _ = self.mp_clip_loss(feats_skel, feats_txt, pos_mask)
            
            # Calculate divergence loss (always active)
            div_loss = self.h_loss(hyper_feats) *0.5 # body part feature divergence loss as cont. loss is not symmetric
            
            l2_penalty_loss = self.logic_layers.l2_penalty()
            l2p_loss = l2_penalty_loss * self.logic_args.l2_weight
            # REG loss: l1 + l2 + entropy loss for logic layers
            # l2p_loss = self.logic_layers.combined_regularization(epoch, total_epochs, warmup_epochs)
            
            # Total loss with conditional logic penalty
            total_loss = (loss_concept * self.args.concept_weight) + \
                        (loss_cont * self.args.contrastive_weight) + \
                        div_loss + action_loss + l2p_loss

        t_concep = torch.sigmoid(concept_logits[:, self.spatial_concepts:])
        s_concep = torch.sigmoid(concept_logits[:, :self.spatial_concepts])
        if self.training:
            return action_probs, s_concep, t_concep, \
                    {
                    'total_loss': total_loss,
                    'action_loss': action_loss,
                    'l2_penalty_loss': l2p_loss,
                    'concept_loss': loss_concept,
                    'contrastive_loss': loss_cont,
                    'div_loss': div_loss,
                    'all_logits': concept_logits,
                    }
        else:
            return action_probs, s_concep, t_concep, {'all_logits': concept_logits}