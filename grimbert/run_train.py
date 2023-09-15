import os
from sacred.experiment import Experiment
from sacred.run import Run
from transformers import BertTokenizerFast
import torch
from grimbert.model import SpeakerAttributionModel
from grimbert.datas import SpeakerAttributionDataset
from grimbert.train import train_speaker_attribution


ROOT_DIR = os.path.dirname(os.path.abspath(f"{__file__}/.."))


ex = Experiment()


@ex.config
def config():

    # see https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/trainer#transformers.TrainingArguments
    hg_training_kwargs: dict

    bert_encoder: str = "bert-base-cased"

    # see :class:`grimbert.model.SpeakerAttributionModelConfig`
    sa_model_config: dict = {"segment_len": 128}

    quote_ctx_len: int = 256


@ex.main
def main(
    _run: Run,
    hg_training_kwargs: dict,
    bert_encoder: str,
    sa_model_config: dict,
    quote_ctx_len: int,
):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SpeakerAttributionDataset.from_muzny_files(
        [f"{ROOT_DIR}/corpus/emma.xml", f"{ROOT_DIR}/corpus/pp.xml"],
        quote_ctx_len,
        sa_model_config["speaker_repr_nb"],
        tokenizer,
    )

    eval_dataset = SpeakerAttributionDataset.from_muzny_files(
        [f"{ROOT_DIR}/corpus/steppe.xml"],
        quote_ctx_len,
        sa_model_config["speaker_repr_nb"],
        tokenizer,
    )

    weights = train_dataset.weights().to(device)
    model = SpeakerAttributionModel.from_pretrained(
        bert_encoder, weights=weights, **sa_model_config
    ).to(device)
    model = train_speaker_attribution(
        model, train_dataset, eval_dataset, _run, **hg_training_kwargs
    )


if __name__ == "__main__":
    ex.run_commandline()
