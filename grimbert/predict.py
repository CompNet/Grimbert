from typing import List, Literal, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from sacremoses import MosesTokenizer
from tqdm import tqdm
from grimbert.datas import (
    SpeakerAttributionDataset,
    SpeakerAttributionDocument,
    DataCollatorForSpeakerAttribution,
    SpeakerAttributionQuote,
    SpeakerAttributionMention,
)
from grimbert.model import SpeakerAttributionModel
from grimbert.utils import find_pattern


@dataclass
class SpeakerPrediction:
    predicted_speaker: Optional[str]
    score: float
    quote: SpeakerAttributionQuote


def predict_speaker(
    dataset: SpeakerAttributionDataset,
    model: SpeakerAttributionModel,
    tokenizer: BertTokenizerFast,
    batch_size: int,
    device: Literal["cuda", "cpu", "auto"] = "auto",
    quiet: bool = False,
) -> List[List[SpeakerPrediction]]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    model = model.eval()
    model = model.to(torch_device)

    data_collator = DataCollatorForSpeakerAttribution(tokenizer)
    dataloader = DataLoader(dataset, batch_size, collate_fn=data_collator)

    preds = [
        [SpeakerPrediction(None, 0.0, quote) for quote in document.quotes]
        for document in dataset.documents
    ]

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=quiet):
            local_batch_size = batch["input_ids"].shape[0]

            batch = {
                k: v.to(torch_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            batch["labels"] = None

            out = model(**batch)
            # (b, 2)
            scores = torch.softmax(out.logits, dim=1)

            for i in range(local_batch_size):
                doc_i = batch["document_i"][i]
                quote_i = batch["quote_i"][i]
                prev_best_score = preds[doc_i][quote_i].score
                score = float(scores[i][1].item())
                if prev_best_score < score:
                    preds[doc_i][quote_i].predicted_speaker = batch["speaker"][i]
                    preds[doc_i][quote_i].score = score
                    preds[doc_i][quote_i].quote.speaker = batch["speaker"][i]

    return preds


def detect_quotes(
    tokens: list[str], quote_pairs: Optional[List[Tuple[str, str]]] = None
) -> list[SpeakerAttributionQuote]:
    """Automatically detect quotes in a sequence of tokens.

    :param tokens: the list of tokens in which to extract quotes.
    :param quote_pairs: a list of quote pairs to detect quotes to be
        used instead of the default.

    :return: a list of quotes
    """
    DEFAULT_QUOTE_PAIRS = [('"', '"'), ("``", "''"), ("«", "»"), ("“", "”")]
    if quote_pairs is None:
        quote_pairs = DEFAULT_QUOTE_PAIRS

    def get_quote_pair(quote: str) -> Optional[Tuple[str, str]]:
        for qp in quote_pairs:
            if quote == qp[0] or quote == qp[1]:
                return qp
        return None

    quotes = []
    cur_quote = None

    for token_i, token in enumerate(tokens):
        if not cur_quote is None:
            cur_quote.tokens.append(token)

        qp = get_quote_pair(token)
        if qp is None:
            continue

        is_opening_quote = token == qp[0]

        if is_opening_quote and cur_quote is None:
            cur_quote = SpeakerAttributionQuote([""], token_i, -1, "")
        else:
            if not cur_quote is None:
                cur_quote.end = token_i + 1
                cur_quote.tokens = tokens[cur_quote.start : cur_quote.end]
                quotes.append(cur_quote)
                cur_quote = None

    return quotes


def predict_speaker_simple(
    text: Union[str, list[str]],
    candidates: list[set[str]],
    model: SpeakerAttributionModel,
    tokenizer: BertTokenizerFast,
    batch_size: int = 1,
    detect_quotes_fn: Callable[
        [list[str]], list[SpeakerAttributionQuote]
    ] = detect_quotes,
    quote_ctx_len: int = 512,
    speaker_repr_nb: int = 4,
) -> list[SpeakerPrediction]:
    """A simple API to predict quote speakers in a text.

    :param text: A text containing quotes for which to predict a
        speaker.
    :param candidates: a list of candidates, each as a set of aliases
        for that candidates.
    :param model: the speaker attribution model to use.
    :param tokenizer: the huggingface tokenizer to use.
    :param batch_size: batch size when predicting.  May be useful in
        case of a large number of speakers.
    :param detect_quotes_fn: function used to detect quote boundaries.
    :param quote_ctx_len: maximum context length, in tokens, to use
    :param speaker_repr_nb: maximum number of speaker representations
        to use.

    :return: A list of speaker prediction, one per quote in the text.
    """
    tokens = text
    if isinstance(tokens, str):
        m_tokenizer = MosesTokenizer(lang="eng")
        tokens = m_tokenizer.tokenize(text, escape=False)

    # collect mentions using the given list of candidate aliases.
    mentions = []
    for aliases in candidates:
        name = list(aliases)[0]
        for alias in aliases:
            if isinstance(alias, str):
                alias = m_tokenizer.tokenize(text, escape=False)
            for start, end in find_pattern(tokens, alias):
                mentions.append(
                    SpeakerAttributionMention(tokens[start:end], start, end, name)
                )

    dataset = SpeakerAttributionDataset(
        [
            SpeakerAttributionDocument(tokens, detect_quotes_fn(tokens), mentions),
        ],
        quote_ctx_len=quote_ctx_len,
        speaker_repr_nb=speaker_repr_nb,
        tokenizer=tokenizer,
    )

    return predict_speaker(dataset, model, tokenizer, batch_size=batch_size)[0]
