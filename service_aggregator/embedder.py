from typing import Callable, List

import fasttext
import numpy as np
import torch
from nltk.tokenize import RegexpTokenizer

ft_model = fasttext.load_model("/Users/malikarii/PycharmProjects/last-work/service_aggregator/models/ru_sentence_embedder_v3.bin")
tokenizer = RegexpTokenizer(r'\w+|[^\w\s]+', gaps=True)
embedder_text = torch.load("/Users/malikarii/PycharmProjects/last-work/service_aggregator/models/ru_sentence_embedder_v4_text.pt")


def preprocess(tokenizer: Callable, text: str) -> List[int]:
    """
    1. Удаляем перенос строки
    2. Переводим в нижний регистр
    3. Бьем на список из слов
    4. Получаем токен для каждого элемента

    Args:
        tokenizer (Callable): токенайзер для текстов
        text (str): Текст для предобработки

    Returns:
        List[int]: Список из токенов
    """
    text = str(text).strip().replace("\n", " ").replace("\xa0", " ").lower()
    tokens = tokenizer.tokenize(text)
    return tokens


def words_to_embed(model: Callable, words) -> np.ndarray:
    """Получаем вектор из токенов для текста.

    Args:
        model (Callable): предобученная модель
        words (_type_): список токенов

    Returns:
        np.ndarray: эмбединг текста
    """

    vectors = [model.get_word_vector(w) for w in words]
    norm_vectors = [x / np.linalg.norm(x) for x in vectors]
    avg_wv = np.mean(norm_vectors, axis=0)
    max_wv = np.max(norm_vectors, axis=0)
    min_wv = np.min(norm_vectors, axis=0)
    return np.concatenate((avg_wv, max_wv, min_wv))


def get_embeddings_text(
    corpus: List[str],
    tokenizer: Callable = tokenizer,
    model_vectorizer: Callable = ft_model,
    model_embedder: Callable = embedder_text,
) -> np.ndarray:
    """Улучшенные векторные представления текстов

    Args:
        model (Callable): предобученная модель
        corpus (List[str]): список текстов для обработки

    Returns:
        List[int]: Список обработанных текстов
    """
    embeddings = list()
    for text in corpus:
        tokens = preprocess(tokenizer=tokenizer, text=text)
        embeddings.append(words_to_embed(model=model_vectorizer, words=tokens))

    embeddings = torch.Tensor(np.array(embeddings))
    improve_embeddings = model_embedder(embeddings)

    return improve_embeddings.tolist()
