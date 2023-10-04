from typing import Optional, Dict
from grimbert.datas import SpeakerAttributionDataset, DataCollatorForSpeakerAttribution
from grimbert.model import SpeakerAttributionModel
from sklearn.metrics import precision_recall_fscore_support
from sacred.run import Run
from transformers import TrainingArguments, Trainer, EvalPrediction
import numpy as np


class SacredTrainer(Trainer):
    def __init__(self, _run: Run, **kwargs):
        super().__init__(**kwargs)
        self._run = _run

    def evaluate(self, **kwargs) -> Dict[str, float]:
        metrics = super().evaluate(**kwargs)
        for k, v in metrics.items():
            self._run.log_scalar(k, v)
        return metrics

    def log(self, logs: Dict[str, float]):
        super().log(logs)
        if "loss" in logs:
            self._run.log_scalar("loss", logs["loss"])
        if "learning_rate" in logs:
            self._run.log_scalar("learning_rate", logs["learning_rate"])


def _train_compute_metrics(eval_prediction: EvalPrediction) -> dict:
    # label_ids   (b, 1)
    # predictions ((b, 2), ...)
    precision, recall, f1, _ = precision_recall_fscore_support(
        eval_prediction.label_ids.flatten(),
        np.argmax(eval_prediction.predictions, axis=1),
        average="binary",
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def train_speaker_attribution(
    model: SpeakerAttributionModel,
    train_dataset: SpeakerAttributionDataset,
    eval_dataset: Optional[SpeakerAttributionDataset] = None,
    _run: Optional[Run] = None,
    **kwargs
) -> SpeakerAttributionModel:
    """
    :param model: model to train
    :param train_dataset:
    :param eval_dataset: evaluation dataset
    :param kwargs: kwargs passed to :class:`transformers.TrainingArguments`
    """
    training_args = TrainingArguments(**kwargs)

    trainer_kwargs = {}
    trainer_class = Trainer

    if not _run is None:
        trainer_class = SacredTrainer
        trainer_kwargs["_run"] = _run

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSpeakerAttribution(train_dataset.tokenizer),
        compute_metrics=_train_compute_metrics,
        **trainer_kwargs
    )
    _ = trainer.train()

    return trainer.model
