import torch
import torch.nn as nn
import torch.utils.checkpoint

import utils as utils


from transformers import T5ForConditionalGeneration, T5Config

import numpy as np


repo = "google-t5/t5-base"

def make_head(inplanes, planes, head_type):
    if head_type == 'linear':
        return nn.Linear(inplanes, planes, bias=False)
    else:
        return nn.Identity()

class TextCLIP(nn.Module):
    def __init__(self, config=None, inplanes=1024, planes=1024, head_type='identy'):
        super(TextCLIP, self).__init__()

        self.model_txt = T5ForConditionalGeneration(config=config).get_encoder() 
        self.lm_head = make_head(self.model_txt.config.d_model, planes, head_type)

    def forward(self, input_ids, attention_mask):
        
        txt_logits = self.model_txt(input_ids=input_ids, attention_mask=attention_mask)[0]
        output = txt_logits[torch.arange(txt_logits.shape[0]), input_ids.argmax(dim=-1)]
        return self.lm_head(output), txt_logits
    
class T5_Model(nn.Module):
    def __init__(self, config=None, inplanes=1024, planes=1024, head_type='linear', lm_head_type='linear'):
        super(T5_Model, self).__init__()
        
        self.model_txt = T5ForConditionalGeneration(config=config)
        if lm_head_type == 'identy':
            self.model_txt.lm_head = nn.Identity()
            self.lm_head = make_head(self.model_txt.config.d_model, planes, head_type)
        self.lm_head_type = lm_head_type

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model_txt(input_ids=input_ids, attention_mask=attention_mask, 
                                    decoder_input_ids=input_ids, 
                                    decoder_attention_mask=attention_mask, **kwargs)
        if self.lm_head_type == 'identy':
            logits = outputs.logits
            output = logits[torch.arange(logits.shape[0]), input_ids.argmax(dim=-1)]
            return self.lm_head(output), logits
        else:
            return outputs
        
    def generate(self, gloss_ids, num_beams):
        return self.model_txt.generate(input_ids=gloss_ids,
                                       num_beams=num_beams)    
    
class GlossTextCLIP(nn.Module):
    def __init__(self, config, embed_dim=1024, task="clip"):
        super(GlossTextCLIP, self).__init__()
        self.task = task
        self.config = T5Config.from_pretrained(config['model']['transformer'], cache_dir="./")
        if self.task == "clip":
            self.model_gloss = T5_Model(self.config, inplanes=embed_dim, planes=embed_dim, lm_head_type="identy")
            self.model_text = TextCLIP(self.config, inplanes=embed_dim, planes=embed_dim, head_type="linear")

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.model_gloss = T5_Model(self.config, inplanes=embed_dim, planes=embed_dim, lm_head_type="linear")
            

    def get_model_txt(self):
        return self.model_text
    
    @property
    def get_encoder_hidden_states(self):
        return self.encoder_hidden_states
    
        
    def forward_clip(self, src_input):
        if torch.cuda.is_available():
            src_input = {k:v.cuda() if isinstance(v, torch.Tensor) else v for k,v in src_input.items()}
            
        
        gloss_features, _ = self.model_gloss(src_input['gloss_ids'], src_input['attention_mask'])
        text_features, self.encoder_hidden_states = self.model_text(src_input['labels'], src_input['labels_attention_mask'])

        gloss_features = gloss_features / gloss_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_gloss = logit_scale * gloss_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ gloss_features.t()

        ground_truth = torch.eye(logits_per_gloss.shape[0], device=logits_per_text.device, dtype=logits_per_gloss.dtype, requires_grad=False)

        return logits_per_gloss, logits_per_text, ground_truth
    
    def forward_g2t(self, src_input):
        
        if torch.cuda.is_available():
            src_input = {k:v.cuda() if isinstance(v, torch.Tensor) else v for k,v in src_input.items()}
        outputs = self.model_gloss(input_ids=src_input['gloss_ids'], 
                                   attention_mask=src_input['attention_mask'],
                                   return_dict=True)
        return outputs.logits

        
    def forward(self, *args, **kwargs):
        if "src_input" in kwargs:
            src_input = kwargs["src_input"]
        else:
            src_input = {
                "gloss_ids" : args[0],
                "attention_mask" : args[1],
                "labels" : args[2],
                "labels_attention_mask" : args[3]
                
            }
        
        src_input["gloss_ids"] = src_input["gloss_ids"].to(torch.long)    
        
        if self.task == "clip":
            return self.forward_clip(src_input)
        elif self.task == "g2t":
            return self.forward_g2t(src_input)
        else:
            raise NotImplementedError(f"Task {self.task} is not supported")
    def generate(self,src_input, num_beams, device=None ):
        out = self.model_gloss.generate(src_input['gloss_ids'].to(device), num_beams=num_beams)
        return out
    
    def __str__(self):
        info = f"""
        Model Information:
        ----------------------
        - Task Type: {self.task}
        - Logit Scale: {'Enabled' if hasattr(self, 'logit_scale') else 'Disabled'}
        - Output Head Type: {self.model_gloss.lm_head_type}
        - Model Mode: {'Training' if self.training else 'Evaluation'}
        """
        return info.strip()
    
