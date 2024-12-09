from turtle import forward
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model, GPT2LMHeadModel
class CustomizedGPT2Attention(GPT2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        # Prepare query, key, value matrix
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)  # (batch_size, seq_len, dim)
        query = self._split_heads(query, self.num_heads, self.head_dim)  # (batch_size, num_heads, seq_len, head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Update cache with previous values if available
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        # Optionally return the cache for reuse
        if use_cache:
            present = (key, value)
        else:
            present = None

        # Self-attention mechanism
        attn_output, _ = self._attn(query, key, value, attention_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)  # (batch_size, seq_len, dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present


class CustomizedGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = CustomizedGPT2Attention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.FloatTensor]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        residual = hidden_states

        # Self-attention
        hidden_states = self.ln_1(hidden_states)
        attn_output, present = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        hidden_states = attn_output + residual

        # Feed-forward
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = feed_forward_hidden_states + residual

        return hidden_states, present


class CustomizedGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([CustomizedGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        assert self._attn_implementation == 'eager', "[NLPDL ERROR] set _attn_implementation to 'eager'"

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]

        # Prepare input embeddings
        inputs_embeds = self.wte(input_ids)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Prepare attention mask
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        # Initialize present key-values cache
        presents = () if use_cache else None
        all_hidden_states = []

        # Iterate over all GPT2 layers
        for i, block in enumerate(self.h):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = block(
                hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                presents = presents + (present,)
            all_hidden_states.append(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
        )


class CustomizedGPT2LMHeadModel(GPT2LMHeadModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomizedGPT2Model(config)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: bool = True,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        return {
            'logits': lm_logits,
            'past_key_values': transformer_outputs.past_key_values,
        }


class PrefixCache:
    """A simple cache for storing past key-values corresponding to input prefixes."""
    def __init__(self):
        self.cache = {}

    def get_cached(self, prefix_ids: Tuple[int]):
        """Retrieve cached past_key_values for a given prefix."""
        return self.cache.get(prefix_ids)

    def add_to_cache(self, prefix_ids: Tuple[int], past_key_values):
        """Store past_key_values for a given prefix."""
        self.cache[prefix_ids] = past_key_values


class CustomizedGPT2LMHeadModelWithCache(CustomizedGPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.prefix_cache = PrefixCache()  # Initialize prefix cache

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: bool = True,
    ):
        batch_size, seq_len = input_ids.shape

        # Split input_ids into prefix (cached) and suffix (to be computed)
        prefix_ids = tuple(input_ids[0, :-1].tolist())  # Assume single batch for simplicity
        suffix_ids = input_ids[:, -1:]

        # Check prefix cache
        cached_past_key_values = self.prefix_cache.get_cached(prefix_ids) if use_cache else None

        # If cache is hit, use cached key-values; otherwise, compute from scratch
        if cached_past_key_values:
            past_key_values = cached_past_key_values
        else:
            # Compute past_key_values for the prefix
            prefix_outputs = self.transformer(
                input_ids=input_ids[:, :-1],
                attention_mask=attention_mask[:, :-1],
                use_cache=use_cache,
            )
            past_key_values = prefix_outputs.past_key_values
            if use_cache:
                self.prefix_cache.add_to_cache(prefix_ids, past_key_values)

        # Continue decoding with suffix
        transformer_outputs = self.transformer(
            input_ids=suffix_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        return {
            'logits': lm_logits,
            'past_key_values': transformer_outputs.past_key_values,
        }
