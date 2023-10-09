from typing import Optional
from transformers import PreTrainedModel
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from grimbert.utils import batch_index_select


class SpeakerAttributionModelConfig(BertConfig):
    def __init__(self, segment_len=512, **kwargs) -> None:
        super().__init__(**kwargs)
        self.segment_len = segment_len


class SpeakerAttributionModel(PreTrainedModel):
    """

    .. note ::

        We use the following short notation to annotate shapes :

        - b: batch_size
        - s: speaker_repr_nb
    """

    config_class = SpeakerAttributionModelConfig

    def __init__(
        self,
        config: SpeakerAttributionModelConfig,
        weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> None:
        super().__init__(config, **kwargs)
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)

        self.linear = torch.nn.Linear(
            2 * self.config.hidden_size + self.config.hidden_size,
            2,
        )

        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        self.post_init()

    def bert_encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param input_ids: (b, q)
        :param attention_mask: (b, q)
        :param token_type_ids: (b, q)

        :return: hidden states of the last layer, of shape (b, q, h)
        """

        # list[(batch_size, <= segment_len, hidden_size)]
        last_hidden_states = []

        def maybe_take_segment(
            tensor: Optional[torch.Tensor], start: int, end: int
        ) -> Optional[torch.Tensor]:
            """
            :param tensor: ``(batch_size, seq_size)``
            """
            return tensor[:, start:end] if not tensor is None else None

        for s_start in range(0, input_ids.shape[1], self.config.segment_len):
            s_end = s_start + self.config.segment_len
            out = self.bert(
                input_ids[:, s_start:s_end],
                attention_mask=maybe_take_segment(attention_mask, s_start, s_end),
                token_type_ids=maybe_take_segment(token_type_ids, s_start, s_end),
                position_ids=maybe_take_segment(position_ids, s_start, s_end),
                head_mask=head_mask,
            )
            last_hidden_states.append(out.last_hidden_state)

        return torch.cat(last_hidden_states, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        quote_span_coords: torch.Tensor,
        speaker_repr_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        **kwargs
    ):
        """
        :param input_ids: (b, q)
        :param attention_mask: (b, q)
        :param token_type_ids: (b, q)
        :param quote_span_coords: (b, 2)
        :param speaker_repr_mask: (b, q)
        :param labels: (b, 1)
        """
        b, q = input_ids.shape
        h = self.config.hidden_size

        encoded = self.bert_encode(input_ids, attention_mask, token_type_ids)
        assert encoded.shape == (b, q, h)

        encoded_quotes = batch_index_select(encoded, 1, quote_span_coords)
        assert encoded_quotes.shape == (b, 2, h)
        encoded_quotes = torch.reshape(encoded_quotes, (b, 2 * h))

        speaker_repr_nb = torch.sum(speaker_repr_mask, dim=1).reshape(b, 1)
        speaker_repr_nb[speaker_repr_nb == 0] = 1
        speaker_repr_mask = torch.stack([speaker_repr_mask] * h, dim=-1)
        speaker_repr = torch.sum(encoded * speaker_repr_mask, dim=1) / speaker_repr_nb
        assert speaker_repr.shape == (b, h)

        scores = self.linear(torch.cat((encoded_quotes, speaker_repr), 1))
        assert scores.shape == (b, 2)

        loss = None
        if not labels is None:
            loss = self.loss_fn(scores, torch.flatten(labels))

        return SequenceClassifierOutput(loss, scores)
