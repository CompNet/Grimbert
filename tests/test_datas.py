import os, glob
from grimbert.datas import SpeakerAttributionDataset
from transformers import BertTokenizerFast


PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__ + "/.."))


def test_load_muzny_dataset():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    dataset = SpeakerAttributionDataset.from_muzny_files(
        glob.glob(f"{PROJECT_ROOT_DIR}/corpus/*.xml"), 256, 4, tokenizer
    )
    assert len(dataset.documents) >= 3
