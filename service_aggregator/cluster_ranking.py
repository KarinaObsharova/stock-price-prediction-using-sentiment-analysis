from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_cluster_score(
    corpus: pd.DataFrame,
    cluster_col: str,
    date_col: str,
) -> pd.Series:
    """Получение метрики ранжирования для каждого кластера.
    Метрика: `возраст_кластера` * `вес кластера`
    Где вес кластера, это количество элементов в нем.

    Args:
        corpus (pd.DataFrame): датафрейм. Обязательно содержит [дата, кластер]
        cluster_col (str): название колонки с метками кластера
        date_col (str): название колонки с датами

    Returns:
        pd.Series: возвращает `series` [кластер - метрика]
    """
    clusters_info = corpus.groupby(cluster_col).agg(
        age=pd.NamedAgg(
            column=date_col, aggfunc=lambda x: (max(x) - min(x)).total_seconds() / 3600
        ),
        weight=pd.NamedAgg(column=cluster_col, aggfunc=lambda x: x.count()),
    )
    if clusters_info["age"].max() != 0.0:
        clusters_info["age"] = 1 - (clusters_info["age"] / clusters_info["age"].max())
    clusters_info["weight"] /= clusters_info["weight"].max()

    clusters_info["score"] = clusters_info["age"] * clusters_info["weight"]

    return clusters_info["score"]


def get_topK_clusters(
    corpus: pd.DataFrame,
    K: int = 3,
    cluster_col: str = "cluster",
    date_col: str = "date",
) -> List[int]:
    """Вернуть список наиболее подходящих по метрике `К` кластеров.
    Метрика: `возраст_кластера` * `вес кластера`
    Где вес кластера, это количество элементов в нем.

    Args:
        corpus (pd.DataFrame): датафрейм. Обязательно содержит [дата, кластер]
        K (int, optional): Количество кластеров на выходе. Defaults to 3.
        cluster_col (str, optional): _название колонки с метками кластера.
            Defaults to "cluster".
        date_col (str, optional): название колонки с датами. Defaults to "date".

    Returns:
        List[int]: возвращает метки наиболее подходящих `К` классов
    """
    scores = get_cluster_score(corpus, cluster_col=cluster_col, date_col=date_col)
    top_K_cluster = scores.sort_values(ascending=False)[:K]
    return top_K_cluster.index


# TODO добавить в скор вес источника (pagerank)
def get_news_score(
    cluster: pd.DataFrame,
    date_col: str,
    embedding_col: str,
) -> pd.Series:
    """Получение метрики ранжирования для каждой новости внутри кластера
    Метрика: `возраст` * `близость`
    Где `возраст` - это возраст новости относительно рождения кластера,
    а `близость` - это средняя близость к другим новостям внутри кластера

    Args:
        cluster (pd.DataFrame): датафрейм новостей конкретного кластера.
            Должен содержать [дата, эмбединг]
        date_col (str): название колонки с датой
        embedding_col (str): название колонки с эмбедингами

    Returns:
        pd.Series: датафрейм с метрикой для каждой новости
    """
    cluster["age"] = (
        cluster[date_col].max() - cluster[date_col]
    ).dt.total_seconds() / 3600
    if cluster["age"].max() != 0.0:
        cluster["age"] = 1 - (cluster["age"] / cluster["age"].max())

    sim = cosine_similarity(cluster[embedding_col].to_list())
    sim = sim[~np.eye(sim.shape[0], dtype=bool)].reshape(sim.shape[0], -1)
    cluster["similarity"] = np.mean(sim, axis=1)

    cluster["score"] = (cluster["age"] * cluster["similarity"]).copy()

    return cluster["score"]


def get_topK_news(
    cluster: pd.DataFrame,
    K: Optional[int] = 3,
    date_col: str = "date",
    embedding_col: str = "embedding",
) -> List[int]:
    """Вернуть список наиболее подходящих по метрике `К` новостей.
    Метрика: `возраст` * `близость`
    Где `возраст` - это возраст новости относительно рождения кластера,
    а `близость` - это средняя близость к другим новостям внутри кластера

    Args:
        cluster (pd.DataFrame): датафрейм новостей конкретного кластера.
            Должен содержать [дата, эмбединг]
        K (int, optional): количество новостей на выходе. Defaults to 3.
        date_col (str, optional): название колонки с датой. Defaults to "date".
        embedding_col (str, optional): название колонки с эмбедингами.
            Defaults to "embedding".

    Returns:
        List[int]: _description_
    """
    scores = get_news_score(cluster, date_col=date_col, embedding_col=embedding_col)
    top_K_news = scores.sort_values(ascending=False)
    if K:
        top_K_news = top_K_news.iloc[:K]
    return top_K_news.index
