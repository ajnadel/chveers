import torch
from transformers import PreTrainedModel, GPT2PreTrainedModel, GPT2Tokenizer
from torch import nn

# The following class is mostly borrowed from Li and Liang's 2021 reference implementation of prefix tuning.
# The original code in available here: https://github.com/XiangLi1999/PrefixTuning
# In particular, `PrefixTuning` class is a heavily simplified version of the similarly-named class
# from the file gpt2/train_control.py within their repo.
class PrefixTuning(GPT2PreTrainedModel):
    """Classification Head for  transformer encoders"""
    def __init__(self, config, model_gpt2, #optim_prefix=False, 
        preseqlen=5, # Number of tokens in the prefix
        deep_param=False
        ):
        super().__init__(config)
        print('under the PrefixTuning model')

        # 1: Init some hyperparameters
        self.match_n_layer = config.n_layer                 # Number of layers
        self.match_n_head = config.n_head                   # Number of attention heads
        self.match_n_embd = config.n_embd // config.n_head  # Embedding dimension PER HEAD
        self.n_embd = config.n_embd                         # Total embedding dimension

        # 2: Take in some other hyperparams

        if hasattr(config, 'preseqlen'):
            self.preseqlen = config.preseqlen
        else:
            self.preseqlen = preseqlen

        self.tuning_mode = 'prefixtune'

        self.task_mode = 'generate'

        if hasattr(config, 'prefix_dropout'):
            self.prefix_dropout = config.prefix_dropout
        else:
            self.prefix_dropout = 0.0

        self.init_random = False

        self.mid_dim = 512


        self.mode_para = 0
        print('PrefixTuning')
        print('preseqlen is {}, optimizing the prefix directly'.format(self.preseqlen))
    
        low_data_init = 0
        print('[Full prefix-tuning Setting :) ]')
        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte = nn.Embedding(self.preseqlen, config.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(config.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, config.n_layer * 2 * config.n_embd))

        self.dropout = nn.Dropout(self.prefix_dropout)
        

    # I think this is the primary one.
    def get_prompt(self, gpt2=None, bsz=None):
        # 1: Get input tokens indices (the prefix/continuous prompt) X = [1,2,3,4,5]
        # 2: Add batch size as a first dimension (unsqueeze & expand)
        # 3: Put on cuda/cpu depending on self.device setting
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        # 4: Map input tokens through word_embeddings
        # this is the part that actually learns the embeddings
        temp_control = self.wte(input_tokens) # X = [1,2,3,4,5] --> [] 
        # do some stuff. feed forward to mid_dim, then apply non-linearity, then back to full
        past_key_values = self.control_trans(temp_control).to(self.device) #bsz, seqlen, layer*emb

        # 6: Get the shape, resize it to prepare for attention!! A-TENNNN-SHUN!
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        # 7: Apply dropout (word to Elizabeth Holmes).
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2) # wht the fuck?
        return past_key_values



    def forward(self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        gpt2_model=None,
        src=None,
        tgt=None,
        src_attn=None,
        tgt_attn=None,
        **kwargs,
        ):

        #{"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn, 'src':src}

        bsz = input_ids.shape[0]

        # Calculate the prefix (this is like a `forward` method for the prompt module)
        past_key_values_prompt = self.get_prompt(gpt2=gpt2_model, bsz=bsz)

        # Make sure that past_key_values is free for our use (to add the prefix)
        if past_key_values is not None:
            assert False, "Attention, use past_key_values for other things"
        else:
            past_key_values = past_key_values_prompt

        # Make sure our gpt model is available
        if gpt2_model is None:
            assert False, "Didn't specify gpt2 model"

        if self.mode_para == 2 and src_attn is not None and tgt_attn is not None:
            attention_mask = torch.cat([src_attn, tgt_attn], dim=1)

        output = gpt2_model(input_ids=input_ids,
                            past_key_values=past_key_values, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                           head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, labels=labels, use_cache=use_cache,
                           output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                           return_dict=return_dict, **kwargs)

        return output