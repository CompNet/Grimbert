from __future__ import annotations
from typing import Any, List, Literal, Optional, Set, Tuple, Dict
from xml.etree import ElementTree as ET
from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from sacremoses import MosesTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from more_itertools import flatten


class DataCollatorForSpeakerAttribution:
    def __init__(self, tokenizer=None) -> None:
        self.tokenizer = tokenizer

    def tok_padding_value(self) -> int:
        if self.tokenizer is None:
            return 0
        return self.tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        b = len(features)
        s = features[0]["speaker_repr_mask"].shape[0]
        q = max([f["input_ids"].shape[0] for f in features])

        batch = {}

        # pad features with variable-length sequence
        for key, padding_value in (
            ("input_ids", self.tok_padding_value()),
            ("attention_mask", 0),
            ("token_type_ids", 0),
            ("speaker_repr_mask", 0),
        ):
            batch[key] = pad_sequence(
                [f[key] for f in features],
                batch_first=True,
                padding_value=padding_value,
            )

        # stack other tensor features
        for key in ("quote_span_coords", "labels"):
            batch[key] = torch.stack([f[key] for f in features], dim=0)

        # concat list features
        for key in ("document_i", "quote_i", "speaker"):
            batch[key] = [f[key] for f in features]

        return batch


@dataclass
class SpeakerAttributionQuote:
    tokens: List[str]
    start: int
    end: int
    speaker: str


@dataclass
class SpeakerAttributionMention:
    tokens: List[str]
    start: int
    end: int
    speaker: str


@dataclass
class SpeakerAttributionDocument:
    tokens: List[str]
    quotes: List[SpeakerAttributionQuote]
    mentions: List[SpeakerAttributionMention]

    @staticmethod
    def dist(quote: SpeakerAttributionQuote, mention: SpeakerAttributionMention) -> int:
        return int(
            abs((quote.start + quote.end) / 2 - (mention.start + mention.end) / 2)
        )

    def speakers(self) -> Set[str]:
        return set(
            [q.speaker for q in self.quotes] + [m.speaker for m in self.mentions]
        )

    def examples(self) -> List[Tuple[SpeakerAttributionQuote, str, bool]]:
        exs = []
        for speaker in sorted(self.speakers()):
            for quote in self.quotes:
                exs.append((quote, speaker, quote.speaker == speaker))
        return exs


class SpeakerAttributionDataset(Dataset):
    def __init__(
        self,
        documents: List[SpeakerAttributionDocument],
        quote_ctx_len: int,
        speaker_repr_nb: int,
        tokenizer: BertTokenizerFast,
    ):
        self.documents = documents
        self.examples = list(
            flatten(
                [
                    [
                        (quote, doc, speaker, label)
                        for quote, speaker, label in doc.examples()
                    ]
                    for doc in documents
                ]
            )
        )
        self.quote_ctx_len = quote_ctx_len
        self.speaker_repr_nb = speaker_repr_nb
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def weights(self) -> torch.Tensor:
        """
        :return: a tensor of shape (2)
        """
        labels = [label for _, _, _, label in self.examples]
        pos_nb = sum(labels)
        neg_nb = len(labels) - pos_nb
        max_nb = max(pos_nb, neg_nb)
        return torch.tensor([max_nb / neg_nb, max_nb / pos_nb])

    @staticmethod
    def from_muzny_files(
        paths: List[str],
        quote_ctx_len: int,
        speaker_repr_nb: int,
        tokenizer: BertTokenizerFast,
    ) -> SpeakerAttributionDataset:
        return SpeakerAttributionDataset(
            list(
                flatten(
                    [
                        SpeakerAttributionDataset._load_muzny_xml_file(path)
                        for path in paths
                    ]
                )
            ),
            quote_ctx_len,
            speaker_repr_nb,
            tokenizer,
        )

    @staticmethod
    def _load_muzny_xml_file(path: str) -> List[SpeakerAttributionDocument]:
        m_tokenizer = MosesTokenizer()

        root = ET.parse(path)

        chapter_nodes = [c for c in root.iter("chapter")]

        documents: List[SpeakerAttributionDocument] = []

        def _parse_xml_(
            node,
            current_i: int,
            document: SpeakerAttributionDocument,
            alias_to_speaker: Dict[str, str],
        ) -> List[str]:
            node.attrib["tokens"] = m_tokenizer.tokenize(
                node.text if not node.text is None else "", escape=False
            )

            child_i = current_i + len(
                m_tokenizer.tokenize(
                    node.text if not node.text is None else "", escape=False
                )
            )
            for child in [c for c in node.findall("./*") if not c is node]:
                try:
                    node.attrib["tokens"] += _parse_xml_(
                        child, child_i, document, alias_to_speaker
                    )
                except TypeError:
                    breakpoint()
                tail_tokens = m_tokenizer.tokenize(
                    child.tail.strip() if not child.tail is None else "", escape=False
                )
                node.attrib["tokens"] += tail_tokens
                child_i += len(child.attrib["tokens"]) + len(tail_tokens)

            tokens = node.attrib["tokens"]
            if node.tag == "quote" and not node.attrib["speaker"] in (
                "NOTANUTTERANCE",
                "UNSURE",
            ):
                document.quotes.append(
                    SpeakerAttributionQuote(
                        tokens,
                        current_i,
                        current_i + len(tokens),
                        alias_to_speaker[node.attrib["speaker"]],
                    )
                )
            elif node.tag == "mention":
                document.mentions.append(
                    SpeakerAttributionMention(
                        tokens,
                        current_i,
                        current_i + len(tokens),
                        alias_to_speaker[node.get("speaker", node.text)],
                    )
                )

            return node.attrib["tokens"]

        # In the muzny dataset, each speaker is mentioned by a bunch
        # of aliases that maps to a single ID
        # { alias_or_id => id }
        alias_to_speaker = {}
        for s_node in root.iter("character"):
            name = s_node.attrib["name"]
            aliases = s_node.attrib["aliases"].split(";")
            assert len(aliases) > 0
            alias_to_speaker[name] = name
            for alias in aliases:
                alias_to_speaker[alias] = name

        for c_node in chapter_nodes:
            document = SpeakerAttributionDocument(
                m_tokenizer.tokenize("".join(c_node.itertext()), escape=False), [], []
            )
            _parse_xml_(c_node, 0, document, alias_to_speaker)
            documents.append(document)

        return documents

    def __getitem__(self, index: int) -> BatchEncoding:

        quote, document, speaker, label = self.examples[index]

        # The whole context used for speaker attribution is of the
        # form [lcontext, quote, rcontext]. The size of lcontext and
        # rcontext depends on the total size allowed for the input,
        # which is self.quote_ctx_len
        quote_outer_ctx_len = (self.quote_ctx_len - (quote.end - quote.start)) // 2
        quote_ctx_start = max(quote.start - quote_outer_ctx_len, 0)
        quote_ctx_end = quote.end + quote_outer_ctx_len
        batch = self.tokenizer(
            document.tokens[quote_ctx_start:quote_ctx_end],
            is_split_into_words=True,
            truncation=True,
            return_tensors="pt",
        )

        for key in batch.keys():
            batch[key] = batch[key][0]

        # function to convert from a position in the initial tokens to
        # a position in the batch
        def pos_to_batchpos(
            pos: int, side: Literal["start", "end"], clip: bool = True
        ) -> Optional[int]:

            pos -= quote_ctx_start

            if pos <= 0:
                return 0 if clip else None
            if pos >= quote_ctx_end - quote_ctx_start:
                return quote_ctx_end - quote_ctx_start - 1 if clip else None

            bpos = batch.word_to_tokens(pos)
            if bpos is None:
                return None
            else:
                return bpos.start if side == "start" else bpos.end - 1

        # quote_span_coords
        quote_b_start = pos_to_batchpos(quote.start, "start")
        quote_b_end = pos_to_batchpos(quote.end - 1, "end")
        batch["quote_span_coords"] = torch.tensor([quote_b_start, quote_b_end])

        # For a given speaker, we need the coordinate of each of its
        # mentions
        speaker_mentions = [m for m in document.mentions if m.speaker == speaker]
        closest_mentions = sorted(
            speaker_mentions, key=lambda m: SpeakerAttributionDocument.dist(quote, m)
        )[: self.speaker_repr_nb]

        speaker_repr_mask = torch.zeros(batch["input_ids"].shape[0])

        for mention in closest_mentions:
            mention_start = pos_to_batchpos(mention.start, "start", clip=False)
            mention_end = pos_to_batchpos(mention.end - 1, "end", clip=False)
            if not mention_start is None and not mention_end is None:
                speaker_repr_mask[mention_start] = 1
                speaker_repr_mask[mention_end] = 1

        batch["speaker_repr_mask"] = speaker_repr_mask

        # Labels
        batch["labels"] = torch.tensor([label]).to(torch.long)

        # Additional infos for inference
        batch["document_i"] = self.documents.index(document)
        batch["quote_i"] = document.quotes.index(quote)
        batch["speaker"] = speaker

        return batch
