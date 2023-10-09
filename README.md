# Grimbert

Speaker attribution in novels. Based on the older [bert-quote-attribution](https://gitlab.com/Aethor/bert-quote-attribution) project.


# Documentation

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
