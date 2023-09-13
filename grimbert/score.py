from typing import List
from grimbert.datas import SpeakerAttributionDataset
from grimbert.predict import SpeakerPrediction


def score_preds(
    dataset: SpeakerAttributionDataset, preds: List[List[SpeakerPrediction]]
) -> float:
    accl = []
    for doc, doc_preds in zip(dataset.documents, preds):
        for quote, pred in zip(doc.quotes, doc_preds):
            accl.append(quote.speaker == pred.predicted_speaker)
    return sum(accl) / len(accl)
