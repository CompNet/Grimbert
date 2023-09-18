from dataclasses import dataclass
from typing import List, Literal, Optional
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from grimbert.datas import SpeakerAttributionDataset, DataCollatorForSpeakerAttribution
from grimbert.model import SpeakerAttributionModel


@dataclass
class SpeakerPrediction:
    predicted_speaker: Optional[str]
    score: float


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
        [SpeakerPrediction(None, 0.0) for _ in document.quotes]
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

    return preds
