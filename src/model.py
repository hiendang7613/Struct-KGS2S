from transformers import T5ForConditionalGeneration
from transformers import T5Config
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from torch import Tensor
import torch.nn.functional as F


def shape(states, batch_size, config):
    return states.view(batch_size, -1, config.num_heads, 64).transpose(1, 2)


class EditedT5v2(T5ForConditionalGeneration):
  def __init__(self, config: T5Config):
    super().__init__(config)

    self.key_projection = nn.Linear(512, config.d_model)
    self.value_projection1 = nn.Linear(512*2, 512*4)
    self.value_projection2 = nn.Linear(512*4, config.d_model)
    self.act = nn.PReLU()
    self.set_config = config
    self.post_init()

  def k_projection(self, x, i, batch_size):
    return shape(self.decoder.block[i].layer[1].EncDecAttention.k(x), batch_size, self.set_config)

  def v_projection(self, x, i, batch_size):
    return shape(self.decoder.block[i].layer[1].EncDecAttention.v(x), batch_size, self.set_config)

  def forward(
      self,
      input_ids: Optional[torch.LongTensor] = None,
      attention_mask: Optional[torch.FloatTensor] = None,
      decoder_input_ids: Optional[torch.LongTensor] = None,
      decoder_attention_mask: Optional[torch.BoolTensor] = None,
      head_mask: Optional[torch.FloatTensor] = None,
      decoder_head_mask: Optional[torch.FloatTensor] = None,
      cross_attn_head_mask: Optional[torch.Tensor] = None,
      encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
      past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
      inputs_embeds: Optional[torch.FloatTensor] = None,
      decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
      labels: Optional[torch.LongTensor] = None,
      use_cache: Optional[bool] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
      target_ent_embeddings=None,
      neighboors_embeddings=None,
      neighboors_embeddings_mask=None,
  ) :

      value_embeddings = self.value_projection2(self.act(self.value_projection1(neighboors_embeddings)))
      key_embeddings = self.key_projection(target_ent_embeddings) * neighboors_embeddings_mask.unsqueeze(2)
      batch_size = value_embeddings.shape[0]

      use_cache = use_cache if use_cache is not None else self.config.use_cache
      return_dict = return_dict if return_dict is not None else self.config.use_return_dict

      if encoder_outputs is None:
        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
      elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
          encoder_outputs = BaseModelOutput(
              last_hidden_state=encoder_outputs[0],
              hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
              attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
          )
      hidden_states = encoder_outputs[0]


      if past_key_values is None:

        past_key_values = []
        for i in range(self.config.num_layers):
          cross_key = self.k_projection(key_embeddings, i, batch_size)
          cross_value = self.v_projection(value_embeddings, i, batch_size)
          self_key = torch.zeros_like(cross_key)
          self_value = torch.zeros_like(cross_value)
          past_key_values.append((self_key, self_value, cross_key, cross_value))


      if self.model_parallel:
          torch.cuda.set_device(self.decoder.first_device)

      if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
          decoder_input_ids = self._shift_right(labels)

      if self.model_parallel:
          torch.cuda.set_device(self.decoder.first_device)
          hidden_states = hidden_states.to(self.decoder.first_device)
          if decoder_input_ids is not None:
              decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
          if attention_mask is not None:
              attention_mask = attention_mask.to(self.decoder.first_device)
          if decoder_attention_mask is not None:
              decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

      # Decode
      decoder_outputs = self.decoder(
          input_ids=decoder_input_ids,
          attention_mask=decoder_attention_mask,
          inputs_embeds=decoder_inputs_embeds,
          past_key_values=past_key_values,
          encoder_hidden_states=hidden_states,
          encoder_attention_mask=attention_mask,
          head_mask=decoder_head_mask,
          cross_attn_head_mask=cross_attn_head_mask,
          use_cache=use_cache,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
      )

      sequence_output = decoder_outputs[0]

      # Set device for model parallelism
      if self.model_parallel:
          torch.cuda.set_device(self.encoder.first_device)
          self.lm_head = self.lm_head.to(self.encoder.first_device)
          sequence_output = sequence_output.to(self.lm_head.weight.device)

      if self.config.tie_word_embeddings:
          sequence_output = sequence_output * (self.model_dim**-0.5)

      lm_logits = self.lm_head(sequence_output)

      loss = None
      if labels is not None:
          loss_fct = CrossEntropyLoss(ignore_index=-100)
          # move labels to correct device to enable PP
          labels = labels.to(lm_logits.device)
          loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

      if not return_dict:
          output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
          return ((loss,) + output) if loss is not None else output

      return Seq2SeqLMOutput(
          loss=loss,
          logits=lm_logits,
          past_key_values=decoder_outputs.past_key_values,
          decoder_hidden_states=decoder_outputs.hidden_states,
          decoder_attentions=decoder_outputs.attentions,
          cross_attentions=decoder_outputs.cross_attentions,
          encoder_last_hidden_state=encoder_outputs.last_hidden_state,
          encoder_hidden_states=encoder_outputs.hidden_states,
          encoder_attentions=encoder_outputs.attentions,
      )

  def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        target_ent_embeddings=None,
        neighboors_embeddings=None,
        neighboors_embeddings_mask=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "target_ent_embeddings": target_ent_embeddings,
            "neighboors_embeddings": neighboors_embeddings,
            "neighboors_embeddings_mask": neighboors_embeddings_mask,
            # "input_ids": input_ids,
        }


def shape(states, batch_size):
    return states.view(batch_size, -1, 8, 64).transpose(1, 2)


class StructKS2S(T5ForConditionalGeneration):
  def __init__(self, config: T5Config):
    super().__init__(config)

    self.key_projection = nn.Linear(700, config.d_model)
    self.value_projection = nn.Linear(700*2, config.d_model)

    self.post_init()

  def k_projection(self, x, i, batch_size):
    return shape(self.decoder.block[i].layer[1].EncDecAttention.k(x), batch_size)

  def v_projection(self, x, i, batch_size):
    return shape(self.decoder.block[i].layer[1].EncDecAttention.v(x), batch_size)

  def forward(
      self,
      input_ids: Optional[torch.LongTensor] = None,
      attention_mask: Optional[torch.FloatTensor] = None,
      decoder_input_ids: Optional[torch.LongTensor] = None,
      decoder_attention_mask: Optional[torch.BoolTensor] = None,
      head_mask: Optional[torch.FloatTensor] = None,
      decoder_head_mask: Optional[torch.FloatTensor] = None,
      cross_attn_head_mask: Optional[torch.Tensor] = None,
      encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
      past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
      inputs_embeds: Optional[torch.FloatTensor] = None,
      decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
      labels: Optional[torch.LongTensor] = None,
      use_cache: Optional[bool] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
      target_ent_embeddings=None,
      neighboors_embeddings=None,
      neighboors_embeddings_mask=None,
  ) :

      value_embeddings = self.value_projection(neighboors_embeddings)
      key_embeddings = self.key_projection(target_ent_embeddings) * neighboors_embeddings_mask.unsqueeze(2)
      value_embeddings = self.value_projection(neighboors_embeddings)
      key_embeddings = self.key_projection(target_ent_embeddings) * neighboors_embeddings_mask.unsqueeze(2)
      batch_size = value_embeddings.shape[0]

      use_cache = use_cache if use_cache is not None else self.config.use_cache
      return_dict = return_dict if return_dict is not None else self.config.use_return_dict

      if encoder_outputs is None:
        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
      elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
          encoder_outputs = BaseModelOutput(
              last_hidden_state=encoder_outputs[0],
              hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
              attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
          )
      hidden_states = encoder_outputs[0]


      if past_key_values is None:

        past_key_values = []
        for i in range(self.config.num_layers):
          cross_key = self.k_projection(key_embeddings, i, batch_size)
          cross_value = self.v_projection(value_embeddings, i, batch_size)
          self_key = torch.zeros_like(cross_key)
          self_value = torch.zeros_like(cross_value)
          past_key_values.append((self_key, self_value, cross_key, cross_value))
          cross_key = self.k_projection(key_embeddings, i, batch_size)
          cross_value = self.v_projection(value_embeddings, i, batch_size)
          self_key = torch.zeros_like(cross_key)
          self_value = torch.zeros_like(cross_value)
          past_key_values.append((self_key, self_value, cross_key, cross_value))


      if self.model_parallel:
          torch.cuda.set_device(self.decoder.first_device)

      if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
          decoder_input_ids = self._shift_right(labels)

      if self.model_parallel:
          torch.cuda.set_device(self.decoder.first_device)
          hidden_states = hidden_states.to(self.decoder.first_device)
          if decoder_input_ids is not None:
              decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
          if attention_mask is not None:
              attention_mask = attention_mask.to(self.decoder.first_device)
          if decoder_attention_mask is not None:
              decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

      # Decode
      decoder_outputs = self.decoder(
          input_ids=decoder_input_ids,
          attention_mask=decoder_attention_mask,
          inputs_embeds=decoder_inputs_embeds,
          past_key_values=past_key_values,
          encoder_hidden_states=hidden_states,
          encoder_attention_mask=attention_mask,
          encoder_hidden_states=hidden_states,
          encoder_attention_mask=attention_mask,
          head_mask=decoder_head_mask,
          cross_attn_head_mask=cross_attn_head_mask,
          use_cache=use_cache,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
      )

      sequence_output = decoder_outputs[0]

      # Set device for model parallelism
      if self.model_parallel:
          torch.cuda.set_device(self.encoder.first_device)
          self.lm_head = self.lm_head.to(self.encoder.first_device)
          sequence_output = sequence_output.to(self.lm_head.weight.device)

      if self.config.tie_word_embeddings:
          sequence_output = sequence_output * (self.model_dim**-0.5)

      lm_logits = self.lm_head(sequence_output)

      loss = None
      if labels is not None:
          loss_fct = CrossEntropyLoss(ignore_index=-100)
          # move labels to correct device to enable PP
          labels = labels.to(lm_logits.device)
          loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

      if not return_dict:
          output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
          return ((loss,) + output) if loss is not None else output

      return Seq2SeqLMOutput(
          loss=loss,
          logits=lm_logits,
          past_key_values=decoder_outputs.past_key_values,
          decoder_hidden_states=decoder_outputs.hidden_states,
          decoder_attentions=decoder_outputs.attentions,
          cross_attentions=decoder_outputs.cross_attentions,
          encoder_last_hidden_state=encoder_outputs.last_hidden_state,
          encoder_hidden_states=encoder_outputs.hidden_states,
          encoder_attentions=encoder_outputs.attentions,
      )

  def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        target_ent_embeddings=None,
        neighboors_embeddings=None,
        neighboors_embeddings_mask=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "target_ent_embeddings": target_ent_embeddings,
            "neighboors_embeddings": neighboors_embeddings,
            "neighboors_embeddings_mask": neighboors_embeddings_mask,
            # "input_ids": input_ids,
            # "input_ids": input_ids,
        }
