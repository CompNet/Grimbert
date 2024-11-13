from __future__ import annotations
from typing import Any, List, Literal, Optional, Set, Tuple, Dict, Union, cast
import glob
from pathlib import Path
from typing_extensions import TypeAlias
from xml.etree import ElementTree as ET
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from transformers.utils import logging as transformers_logging
from tqdm import tqdm
from sacremoses import MosesTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from more_itertools import flatten, windowed


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

    def ctx_bounds(self, quote_ctx_len: int) -> Tuple[int, int]:
        quote_outer_ctx_len = (quote_ctx_len - (self.end - self.start)) // 2
        quote_ctx_start = max(self.start - quote_outer_ctx_len, 0)
        quote_ctx_end = self.end + quote_outer_ctx_len
        return (quote_ctx_start, quote_ctx_end)


@dataclass
class SpeakerAttributionMention:
    tokens: List[str]
    start: int
    end: int
    speaker: str


class SpeakerAttributionDocument:
    def __init__(
        self,
        tokens: List[str],
        quotes: List[SpeakerAttributionQuote],
        mentions: List[SpeakerAttributionMention],
    ):
        self.tokens = tokens
        self.quotes = sorted(quotes, key=lambda q: q.start)
        self.mentions = sorted(mentions, key=lambda m: m.start + (m.end - m.start) / 2)

    @staticmethod
    def dist(quote: SpeakerAttributionQuote, mention: SpeakerAttributionMention) -> int:
        if mention.start < quote.start:
            return quote.start - mention.start
        if mention.end > quote.end:
            return mention.end - quote.end
        return 0

    def speakers(self) -> Set[str]:
        return set(
            [q.speaker for q in self.quotes] + [m.speaker for m in self.mentions]
        )

    @staticmethod
    def mentions_in_range(
        quote: SpeakerAttributionQuote,
        mentions: List[SpeakerAttributionMention],
        quote_ctx_len: int,
    ) -> List[SpeakerAttributionMention]:
        """
        :param mentions: *sorted* list of mentions
        """
        outer_ctx_len = (quote_ctx_len - (quote.end - quote.start)) // 2

        in_range_mentions = []
        for mention in mentions:
            if SpeakerAttributionDocument.dist(quote, mention) <= outer_ctx_len:
                in_range_mentions.append(mention)
            # since mentions are sorted left-to-right, we can early
            # return here: the distance is greater than outer_ctx_len,
            # and the next mentions will have greater distances
            elif mention.start > quote.start:
                break

        return in_range_mentions

    def examples(
        self, quote_ctx_len: int
    ) -> List[Tuple[SpeakerAttributionQuote, str, bool]]:
        exs = []
        for speaker in self.speakers():
            speaker_mentions = [m for m in self.mentions if m.speaker == speaker]
            for quote in self.quotes:
                mentions_in_range = self.mentions_in_range(
                    quote, speaker_mentions, quote_ctx_len
                )
                if len(mentions_in_range) > 0:
                    exs.append((quote, speaker, quote.speaker == speaker))
        return exs


SpeakerAttributionExample: TypeAlias = Tuple[
    SpeakerAttributionQuote, SpeakerAttributionDocument, str, bool
]


class SpeakerAttributionDataset(Dataset):
    def __init__(
        self,
        documents: List[SpeakerAttributionDocument],
        quote_ctx_len: int,
        speaker_repr_nb: int,
        tokenizer: BertTokenizerFast,
        _examples: Optional[List[List[SpeakerAttributionExample]]] = None,
    ):
        """
        :param quote_ctx_len: total size of the quote context
            (including the quote), in tokens (not wordpieces)
        """
        self.documents = documents

        if _examples:
            self.examples = _examples
        else:
            self.examples = []
            for doc in tqdm(self.documents, desc="generating examples"):
                self.examples.append(
                    [
                        (quote, doc, speaker, label)
                        for quote, speaker, label in doc.examples(quote_ctx_len)
                    ]
                )

        self.quote_ctx_len = quote_ctx_len
        self.speaker_repr_nb = speaker_repr_nb
        self.tokenizer = tokenizer

    def __len__(self):
        return len(list(flatten(self.examples)))

    def example_at(self, example_i: int) -> SpeakerAttributionExample:
        return list(flatten(self.examples))[example_i]

    def pprint_example_at(self, example_i: int):
        from rich.console import Console
        from rich.text import Text

        quote, doc, candidate_speaker, label = self.example_at(example_i)

        label_color = "green" if label else "red"

        quote_ctx_start, quote_ctx_end = quote.ctx_bounds(self.quote_ctx_len)

        stylized_tokens: List[Any] = [
            f"{t} " for t in doc.tokens[quote_ctx_start:quote_ctx_end]
        ]

        start = quote.start - quote_ctx_start
        end = quote.end - quote_ctx_start
        for quote_i in range(start, end):
            token = stylized_tokens[quote_i]
            stylized_tokens[quote_i] = (token, "bold magenta")

        speaker_mentions = doc.mentions_in_range(
            quote,
            [m for m in doc.mentions if m.speaker == candidate_speaker],
            self.quote_ctx_len,
        )
        for mention in speaker_mentions:
            start = mention.start - quote_ctx_start
            end = mention.end - quote_ctx_start
            for mention_i in range(start, end):
                token = stylized_tokens[mention_i]
                if isinstance(token, tuple):
                    token = token[0]
                stylized_tokens[mention_i] = (token, f"bold {label_color}")

        text = Text.assemble(*stylized_tokens)
        console = Console(color_system="standard")
        console.print(
            f"candidate speaker: [{label_color}]'{candidate_speaker}'[/{label_color}]"
        )
        console.print(text)

    def splitted(
        self, ratio: float
    ) -> Tuple[SpeakerAttributionDataset, SpeakerAttributionDataset]:
        limit = int(len(self.documents) * ratio)
        return (
            SpeakerAttributionDataset(
                self.documents[:limit],
                self.quote_ctx_len,
                self.speaker_repr_nb,
                self.tokenizer,
                _examples=self.examples[:limit],
            ),
            SpeakerAttributionDataset(
                self.documents[limit:],
                self.quote_ctx_len,
                self.speaker_repr_nb,
                self.tokenizer,
                _examples=self.examples[limit:],
            ),
        )

    def weights(self) -> torch.Tensor:
        """
        :return: a tensor of shape (2)
        """
        labels = [label for _, _, _, label in flatten(self.examples)]
        pos_nb = sum(labels)
        neg_nb = len(labels) - pos_nb
        max_nb = max(pos_nb, neg_nb)
        return torch.tensor([max_nb / neg_nb, max_nb / pos_nb])

    @staticmethod
    def from_PDNC(
        root: Union[str, Path],
        quote_ctx_len: int,
        speaker_repr_nb: int,
        tokenizer: BertTokenizerFast,
    ) -> SpeakerAttributionDataset:
        if isinstance(root, str):
            root = Path(root)

        documents = [
            SpeakerAttributionDataset._load_PDNC_book(Path(path))
            for path in tqdm(
                sorted(glob.glob(f"{root}/data/*")), desc="loading documents"
            )
        ]
        return SpeakerAttributionDataset(
            documents, quote_ctx_len, speaker_repr_nb, tokenizer
        )

    @staticmethod
    def _load_PDNC_book(root: Path) -> SpeakerAttributionDocument:

        quotation_info = pd.read_csv(root / "quotation_info.csv")

        with open(root / "novel_text.txt") as f:
            text = f.read()

        m_tokenizer = MosesTokenizer()

        # Extract tokens and quotes
        # -------------------------
        # In the dataframe, each line can contain "subquotes". We do not
        # support that concept, so we map each subquote to a quote by
        # 'flattening' the structure of the dataframe.
        quote_texts = []
        quote_spans = []
        quote_speakers = []
        for _, qline in quotation_info.iterrows():
            spans = eval(qline["quoteByteSpans"])
            for span in spans:
                # we take into account the quotation marks, something that is
                # not done in the original dataset
                quote_spans.append((span[0] - 1, span[1] + 1))
                quote_speakers.append(qline["speaker"])
                quote_text = text[span[0] - 1 : span[1] + 1]
                quote_texts.append(quote_text)

        # Init doc_tokens with the tokens from the start of the text to the first quote
        doc_tokens: List[str] = m_tokenizer.tokenize(
            text[: quote_spans[0][0]], escape=False
        )
        quotes = []

        # Parse the flattened structure into SpeakerAttributionQuote, keeping
        # track of token indices
        for i, (quote_text, span, speaker) in enumerate(
            zip(quote_texts, quote_spans, quote_speakers)
        ):
            quote_tokens = m_tokenizer.tokenize(quote_text, escape=False)
            quotes.append(
                SpeakerAttributionQuote(
                    quote_tokens,
                    len(doc_tokens),
                    len(doc_tokens) + len(quote_tokens),
                    speaker,
                )
            )
            doc_tokens += quote_tokens

            # Add tokens from the text in between quotes (or the from the text
            # from the last quote to the end of the text)
            if i + 1 < len(quotation_info):
                next_span = quote_spans[i + 1]
                in_between_tokens = m_tokenizer.tokenize(
                    text[span[1] : next_span[0]], escape=False
                )
                doc_tokens += in_between_tokens
            else:
                end_tokens = m_tokenizer.tokenize(
                    text[quote_spans[-1][1] :], escape=False
                )
                doc_tokens += end_tokens

        # Extract mentions
        # ----------------
        characters_info = pd.read_csv(root / "character_info.csv")

        # { alias => normalized name }
        alias_to_speaker = {}
        for _, row in characters_info.iterrows():
            speaker = row["Main Name"]
            speaker_key = tuple(m_tokenizer.tokenize(row["Main Name"], escape=False))
            alias_to_speaker[speaker_key] = speaker
            for alias in eval(row["Aliases"]):
                alias = tuple(m_tokenizer.tokenize(alias, escape=False))
                alias_to_speaker[alias] = speaker

        longest_alias_len = max([len(alias) for alias in alias_to_speaker.keys()])
        mentions = []
        visited_patterns = np.zeros((len(doc_tokens),))

        # iterate from largest pattern to smaller ones. Priority is
        # given to larger patterns, to avoid situations where a
        # smaller pattern disallows matching a larger one. This can
        # happen in the case of charaters sharing a family name: in
        # 'Zoé Traitor', the 'Traitor' part of the name could be
        # matched to the character 'John Traitor' if John Traitor has
        # a 'Traitor' alias.
        for pattern_len in range(longest_alias_len, 0, -1):

            for pattern_i, pattern in enumerate(windowed(doc_tokens, pattern_len)):

                if tuple(pattern) in alias_to_speaker:

                    start = pattern_i
                    end = pattern_i + len(pattern)

                    # check if the current pattern overlaps with a
                    # larger pattern assigned to another speaker. In
                    # that case, we drop the current pattern: we cant
                    # have overlapping speaker representations.
                    if sum(visited_patterns[start:end]) >= 1:
                        continue

                    speaker = alias_to_speaker[tuple(pattern)]
                    mention = SpeakerAttributionMention(
                        list(pattern), pattern_i, end, speaker  # type: ignore
                    )
                    mentions.append(mention)
                    visited_patterns[start:end] = 1

        # common sense check
        assert len(mentions) > 0

        # we're done!
        return SpeakerAttributionDocument(doc_tokens, quotes, mentions)

    @staticmethod
    def from_muzny_files(
        paths: List[str],
        quote_ctx_len: int,
        speaker_repr_nb: int,
        tokenizer: BertTokenizerFast,
        use_additional_gold_mentions: bool = True,
    ) -> SpeakerAttributionDataset:
        """
        :param use_additional_gold_mentions: if ``True``, use mentions
            attributed to a speaker in addition to mentions obtained
            with aliases.
        """
        return SpeakerAttributionDataset(
            list(
                flatten(
                    [
                        SpeakerAttributionDataset._load_muzny_xml_file(
                            path, use_additional_gold_mentions
                        )
                        for path in paths
                    ]
                )
            ),
            quote_ctx_len,
            speaker_repr_nb,
            tokenizer,
        )

    @staticmethod
    def _load_muzny_xml_file(
        path: str, use_additional_gold_mentions: bool
    ) -> List[SpeakerAttributionDocument]:
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

                # Some mentions have a 'speaker' annotation indicating
                # the refered characters
                # example : <mention speaker="Mr_Henry_Woodhouse">papa</mention>
                # we use these only if use_additional_gold_mentions is passed
                if use_additional_gold_mentions:
                    speaker = alias_to_speaker[node.get("speaker", node.text)]
                else:
                    speaker = alias_to_speaker.get(node.text)

                if not speaker is None:
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
                m_tokenizer.tokenize("".join(c_node.itertext()), escape=False),
                [],
                [],
            )
            _parse_xml_(c_node, 0, document, alias_to_speaker)
            documents.append(document)

        return documents

    def __getitem__(self, index: int) -> BatchEncoding:

        quote, document, speaker, label = self.example_at(index)

        # The whole context used for speaker attribution is of the
        # form [lcontext, quote, rcontext]. The size of lcontext and
        # rcontext depends on the total size allowed for the input,
        # which is self.quote_ctx_len
        quote_ctx_start, quote_ctx_end = quote.ctx_bounds(self.quote_ctx_len)
        # NOTE: we disable tokenizer warning to avoid a length
        # ----  warning. Usually, sequences should be truncated to a max
        #       length (512 for BERT). However, in our case, the sequence is
        #       later cut into segments of configurable size, so this does
        #       not apply
        transformers_logging.set_verbosity_error()
        batch = self.tokenizer(
            document.tokens[quote_ctx_start:quote_ctx_end],
            is_split_into_words=True,
            truncation=False,
            return_tensors="pt",
        )
        transformers_logging.set_verbosity_info()

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
                if clip:
                    print("[warning] quote is too big for batch")
                    return 0 if side == "start" else quote_ctx_end - quote_ctx_start - 1
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
