"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
import copy
import json
import logging
from io import open
import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.roberta.modeling_roberta import RobertaPooler
# from apex.normalization.fused_layer_norm import FusedLayerNorm

from layer import BertLayer, BertPooler


logger = logging.getLogger(__name__)


class GELU(nn.Module):
    def forward(self, input_):
        output = self.gelu(input_)
        return output

    def gelu(self, x):
        """Implementation of the gelu activation function.
            For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    Args:
        x: torch.Tensor x:
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class UniterConfig(object):
    """Configuration class to store the configuration of a `UniterModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs UniterConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
                `UniterModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer
                encoder.
            num_attention_heads: Number of attention heads for each attention
                layer in the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e.
                feed-forward) layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string)
                in the encoder and pooler. If string, "gelu", "relu" and
                "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully
                connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this
                model might ever be used with. Typically set this to something
                large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed
                into `UniterModel`.
            initializer_range: The sttdev of the truncated_normal_initializer
                for initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file,
                      "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `UniterConfig` from a
           Python dictionary of parameters."""
        config = UniterConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `UniterConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class UniterPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, UniterConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `UniterConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        # elif isinstance(module, FusedLayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, config_file, state_dict=None, *inputs, **kwargs):
        """
        Instantiate a UniterPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific Uniter class
        """
        # Load config
        if state_dict is None:
            state_dict = {}
        config = UniterConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None
                              else metadata.get(prefix[:-1], {}))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.')
                                              for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from "
                        "pretrained model: {}".format(
                            model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in "
                        "{}: {}".format(
                            model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for '
                               '{}:\n\t{}'.format(
                                   model.__class__.__name__,
                                   "\n\t".join(error_msgs)))
        return model


class ImageRobertaTextEmbeddings(nn.Module):
    """
    Same as RobertaEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_roberta.RobertaEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None,
                inputs_embeds=None, past_key_values_length=0):

        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings                               # (b, x, 768)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        Args:
            inputs_embeds: torch.Tensor
        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class ImageRobertaImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.pos_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.pos_linear = nn.Linear(7, config.hidden_size)                  # (x1, y1, x2, y2, w, h, w * h)
        self.person_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.person_linear = nn.Linear(config.vocab_size, config.hidden_size)
        self.person_layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_feat, person_ids, type_embeddings, img_masks=None):
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_layer_norm(self.img_linear(img_feat.to(torch.float32)))  # (b, 1, 768)
        embeddings = transformed_im + type_embeddings
        if img_pos_feat is not None:
            transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
            embeddings += transformed_pos

        if person_ids is not None:
            transformed_person = self.person_layer_norm(self.person_embed(person_ids))
            embeddings += transformed_person

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, input_, attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        hidden_states = input_
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class ImageRobertaModel(UniterPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.embeddings = ImageRobertaTextEmbeddings(config)
        self.img_embeddings = ImageRobertaImageEmbeddings(config, img_dim)
        self.encoder = UniterEncoder(config)
        self.pooler = RobertaPooler(config)
        self.apply(self.init_weights)

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        output = self.embeddings(flat_input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, person_ids, img_masks=None,
                                img_type_ids=None):
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        img_type_embeddings = self.embeddings.token_type_embeddings(
            img_type_ids)
        output = self.img_embeddings(img_feat, img_pos_feat, person_ids,
                                     img_type_embeddings, img_masks)
        return output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_pos_feat, person_ids,
                                    gather_index, img_masks=None,
                                    txt_type_ids=None, img_type_ids=None):
        txt_emb = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(img_feat, img_pos_feat, person_ids, img_masks, img_type_ids)

        embedding_output = torch.cat([txt_emb, img_emb], dim=1)

        return embedding_output

    def forward(self, input_ids, position_ids, attention_mask, img_feat,
                img_pos_feat, person_ids, gather_index, img_masks=None,
                txt_type_ids=None, img_type_ids=None):

        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(img_feat, img_pos_feat, person_ids, img_masks, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
        else:
            embedding_output = self._compute_img_txt_embeddings(input_ids, position_ids, img_feat, img_pos_feat,
                                                                person_ids, gather_index, img_masks,
                                                                txt_type_ids, img_type_ids)

        # compute self-attention mask
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        extended_attention_mask = flat_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(embedding_output, extended_attention_mask)

        sequence_output = encoded_layers[0]
        pooled_output = self.pooler(sequence_output)

        return pooled_output


class ImageRobertaForMultipleChoice(UniterPreTrainedModel):

    def __init__(self, config, img_dim=512, num_answer=1):
        super().__init__(config)
        self.transformer = ImageRobertaModel(config, img_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            nn.LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, num_answer)
        )
        self.apply(self.init_weights)

    def forward(self, input_ids, img_feat=None, attention_mask=None, labels=None, position_ids=None,
                img_pos_feat=None, person_ids=None, gather_index=None):
        pooled_output = self.transformer(input_ids, position_ids, attention_mask, img_feat,
                                         img_pos_feat, person_ids, gather_index)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, 4)

        loss = F.cross_entropy(reshaped_logits, labels.long())
        correct = (torch.sum(torch.eq(torch.argmax(F.softmax(reshaped_logits, dim=1), dim=1), labels)))
        return dict(
            logits=logits,
            loss=loss,
            correct=correct
        )




