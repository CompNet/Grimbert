import os
from typing import Literal, Optional
from sacred.experiment import Experiment
from sacred.run import Run
from transformers import BertTokenizerFast
import torch
from grimbert.model import SpeakerAttributionModel
from grimbert.datas import SpeakerAttributionDataset
from grimbert.train import train_speaker_attribution
from grimbert.predict import predict_speaker
from grimbert.score import score_preds


ROOT_DIR = os.path.dirname(os.path.abspath(f"{__file__}/.."))


ex = Experiment()


@ex.config
def config():

    # see https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/trainer#transformers.TrainingArguments
    hg_training_kwargs: dict

    bert_encoder: str = "bert-base-cased"

    # which corpus to use. Either "muzny" or "PDNC"
    corpus_name: str = "muzny"
    corpus_path = None
    # additional kwargs, depending on corpus
    corpus_kwargs: dict = {}

    # see :class:`grimbert.model.SpeakerAttributionModelConfig`
    sa_model_config: dict = {"segment_len": 512}

    quote_ctx_len: int = 256
    speaker_repr_nb: int = 4


@ex.main
def main(
    _run: Run,
    hg_training_kwargs: dict,
    bert_encoder: str,
    corpus_name: Literal["muzny", "PDNC"],
    corpus_path: Optional[str],
    corpus_kwargs: dict,
    sa_model_config: dict,
    quote_ctx_len: int,
    speaker_repr_nb: int,
):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if corpus_name == "muzny":
        use_agm = corpus_kwargs.get("use_additional_gold_mentions", False)
        train_dataset = SpeakerAttributionDataset.from_muzny_files(
            [f"{ROOT_DIR}/corpus/emma.xml", f"{ROOT_DIR}/corpus/pp.xml"],
            quote_ctx_len,
            speaker_repr_nb,
            tokenizer,
            use_additional_gold_mentions=use_agm,
        )
        eval_dataset = SpeakerAttributionDataset.from_muzny_files(
            [f"{ROOT_DIR}/corpus/steppe.xml"],
            quote_ctx_len,
            speaker_repr_nb,
            tokenizer,
            use_additional_gold_mentions=use_agm,
        )
    elif corpus_name == "PDNC":
        assert not corpus_path is None
        dataset = SpeakerAttributionDataset.from_PDNC(
            corpus_path, quote_ctx_len, speaker_repr_nb, tokenizer
        )
        train_dataset, eval_dataset = dataset.splitted(0.8)
    else:
        raise ValueError(f"unknown corpus: {corpus_name}")

    weights = train_dataset.weights().to(device)
    model = SpeakerAttributionModel.from_pretrained(
        bert_encoder, weights=weights, **sa_model_config
    ).to(device)
    model = train_speaker_attribution(
        model, train_dataset, eval_dataset, _run, **hg_training_kwargs
    )

    eval_batch_size = hg_training_kwargs.get("per_device_eval_batch_size", 1)
    preds = predict_speaker(eval_dataset, model, tokenizer, eval_batch_size, "auto")
    accuracy = score_preds(eval_dataset, preds)

    print(f"{accuracy=}")

    _run.log_scalar("accuracy", accuracy)


if __name__ == "__main__":
    ex.run_commandline()
