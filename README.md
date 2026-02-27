# Grimbert

Speaker attribution in novels. Based on the older [bert-quote-attribution](https://gitlab.com/Aethor/bert-quote-attribution) project.


# Documentation

The high level API is simple to use:

```python
from grimbert.model import SpeakerAttributionModel
from transformers import BertTokenizerFast
from grimbert.predict import predict_speaker_simple


model = SpeakerAttributionModel.from_pretrained(
	"compnet-renard/spanbert-base-cased-literary-speaker-attribution"
)
tokenizer = BertTokenizerFast.from_pretrained(
	"compnet-renard/spanbert-base-cased-literary-speaker-attribution"
)

speakers = predict_speaker_simple(
    '"This is horrible", John said to Max', # input text
    [{"John"}, {"Max"}],                    # speaker candidates
    model,
    tokenizer
)
print(speakers)
# [SpeakerPrediction(predicted_speaker='John', score=0.6657354831695557, quote=SpeakerAttributionQuote(tokens=['"', 'This', 'is', 'horrible', '"'], start=0, end=5, speaker='John'))]
```

## Training

It is possible to train a model on the QuoteLi3 dataset from Muzny et al. or on PDNC with the `grimbert.run_train.py` training script. For example:

```sh
python grimbert/run_train.py with hg_training_kwargs='{"learning_rate": 5e5, "num_train_epochs": 5}' bert_encoder='answerdotai/ModernBERT-base' corpus_name='PDNC' corpus_path=/path/to/PDNC corpus_kwargs='{}' sa_model_config='{"segment_len": 512}' quote_ctx_len=512 speaker_repr_nb=4
```

See the training script in itself for more details about parameters.

## Low-level API

There is a more difficult to use lower level prediction API if necessary:

```python
from grimbert.model import SpeakerAttributionModel
from grimbert.predict import predict_speaker
from grimbert.datas import (
    SpeakerAttributionDataset,
    SpeakerAttributionDocument,
    SpeakerAttributionQuote,
    SpeakerAttributionMention
) 
from transformers import BertTokenizerFast


model = SpeakerAttributionModel.from_pretrained(
	"compnet-renard/spanbert-base-cased-literary-speaker-attribution"
)
tokenizer = BertTokenizerFast.from_pretrained(
	"compnet-renard/spanbert-base-cased-literary-speaker-attribution"
)

tokens = '" This is horrible " , John said to Max .'.split(" ")
quote_start = 0
quote_end = 4
john_mention_start = 6
john_mention_end = 7
max_mention_start = 9
max_mention_end = 10

dataset = SpeakerAttributionDataset(
    [
        SpeakerAttributionDocument(
            tokens,
            [SpeakerAttributionQuote(
                tokens[quote_start:quote_end], quote_start, quote_end, "John"
            )],
            [
                SpeakerAttributionMention(
                    tokens[john_mention_start:john_mention_end],
                    john_mention_start,
                    john_mention_end,
                    "John"
                ),
                SpeakerAttributionMention(
                    tokens[max_mention_start:max_mention_end],
                    max_mention_start,
                    max_mention_end,
                    "Max"
                ),
            ]
            
        )
    ],
    quote_ctx_len=512,
    speaker_repr_nb=4, 
    tokenizer=tokenizer
)

preds = predict_speaker(dataset, model, tokenizer, batch_size=4)
```
