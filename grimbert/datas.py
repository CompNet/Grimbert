from __future__ import annotations
from typing import Any, List, Literal, Optional, Set, Tuple, Dict, Union
import glob
from pathlib import Path
from xml.etree import ElementTree as ET
from dataclasses import dataclass
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from tqdm import tqdm
from sacremoses import MosesTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from more_itertools import flatten
from grimbert.utils import find_pattern


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

    def splitted(
        self, ratio: float
    ) -> Tuple[SpeakerAttributionDataset, SpeakerAttributionDataset]:
        return (
            SpeakerAttributionDataset(
                self.documents[: int(len(self.documents) * ratio)],
                self.quote_ctx_len,
                self.speaker_repr_nb,
                self.tokenizer,
            ),
            SpeakerAttributionDataset(
                self.documents[int(len(self.documents) * ratio) :],
                self.quote_ctx_len,
                self.speaker_repr_nb,
                self.tokenizer,
            ),
        )

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
            for path in tqdm(sorted(glob.glob(f"{root}/data/*")))
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
        doc_tokens = m_tokenizer.tokenize(text[: quote_spans[0][0]], escape=False)
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
            alias_to_speaker[row["Main Name"]] = row["Main Name"]
            for alias in eval(row["Aliases"]):
                alias_to_speaker[alias] = row["Main Name"]

        # extract mentions from doc_tokens
        mentions = []
        for alias, speaker in alias_to_speaker.items():
            alias_tokens = m_tokenizer.tokenize(alias, escape=False)
            coords_lst = find_pattern(doc_tokens, alias_tokens)
            for start, end in coords_lst:
                mentions.append(
                    SpeakerAttributionMention(alias_tokens, start, end, speaker)
                )

        # we're done!
        return SpeakerAttributionDocument(doc_tokens, quotes, mentions)

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
