from typing import List

from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

import pdb
import fasttext


tokenizer = RegexTokenizer()
model = FastTextSocialNetworkModel(tokenizer=tokenizer)


def get_sentiment(text: str) -> List[int]:
    print(text)
    results = model.predict(text, k=3)
    sentiments = []
    print(results)
    for result in results:
        key, value = next(iter(result.items()))
        if key == 'neutral':
            sentiments.append(0)
        elif key == 'positive':
            sentiments.append(1)
        elif key == 'negative':
            sentiments.append(-1)
        else:
            sentiments.append(None)
    return sentiments













