import pdb
from typing import Dict, List, Optional, Tuple
import pandas as pd
from service_repository.crud import get_corpus
from sqlalchemy.orm import Session, sessionmaker
from service_aggregator import get_clusters, get_topK_clusters, get_topK_news
from service_aggregator.text_tone_analyzer import get_sentiment
from service_repository.database import engine
from service_repository.dependencies import get_db
from service_repository.models import NewsEmbModel, NewsModel

CORPUS_COLS = ["url", "title"]

def prepare_corpus(
        corpus: List[Tuple[NewsModel, NewsEmbModel]], cols: List[str] = None) -> pd.DataFrame:
    """Преобразование корпуса текстов из orm моделей в pandas DF

    Args:
        corpus (List[Tuple[NewsModel, NewsEmbModel]]): корпус текстов из БД
        cols (List[str]): Столбцы, с которыми вернем DF

    Returns:
        pd.DataFrame: Корпус новостей
    """
    cols = cols or CORPUS_COLS
    corpus_sel_col = [{"url": news.url, "title": news.title} for news in corpus]
    return pd.DataFrame(corpus_sel_col, columns=cols)

def run_pipeline(
        embedding_col: str,
        date_col: str,
        corpus: pd.DataFrame,
        clusters: Optional[int]
) -> List[Dict[int, List[dict]]]:
    """Запуск пайплайна выделения кластеров и отбора новостей в кластере.

    Args:
        corpus (pd.DataFrame): датафрейм новостей.
            Должен содержать [дата, эмбединг]
        clusters (int): количество новостей на выходе.
        date_col (str, optional): название колонки с датой.
        embedding_col (str, optional): название колонки с эмбедингами.

    Returns:
        Dict[int, List[dict]] : кластеризованный корпус
    """
    corpus["cluster"] = get_clusters(corpus[embedding_col].to_list())
    clusters_index = get_topK_clusters(corpus, K=clusters, date_col=date_col)
    clustered_corpus = []
    for cluster in clusters_index:
        corpus_cluster = corpus[corpus["cluster"] == cluster]
        news_index = get_topK_news(
            corpus_cluster,
            date_col=date_col,
            embedding_col=embedding_col,
        )
        news_top = corpus_cluster.loc[news_index]
        clustered_corpus.append(
            {
                "trand_id": cluster,
                "trand_title": news_top.loc[news_index[0], "title"],
                "news": news_top.to_dict(orient="records"),
            }
        )
    return clustered_corpus

def save_sentiment_to_db(db: Session, corpus_with_sentiment: pd.DataFrame):
    """Сохранение результатов анализа тональности в базу данных"""
    for index, row in corpus_with_sentiment.iterrows():
        news = db.query(NewsModel).filter_by(url=row['url']).first()
        if news:
            news.label = row['sentiment']
            db.add(news)
    db.commit()

def get_corpus_for_clusterization(
    num_trands: int,
    db: Session = None,
):
    """Выгрузка актуальных новостей.

    Возможно конфигурировать:
    - num_trands - кол-во трендов

    """
    if db is None:
        db = get_db()
    corpus = get_corpus(db)
    corpus = prepare_corpus(corpus)
    clustered_corpus = run_pipeline(
        corpus=corpus,
        clusters=num_trands,
        embedding_col="embedding_full_text",
        date_col="post_dttm",
    )

    return clustered_corpus

def add_sentiment_to_corpus(corpus: pd.DataFrame) -> pd.DataFrame:
    """Добавление столбца с тональностью к заголовку новостей.

    Args:
        corpus (pd.DataFrame): Корпус новостей.

    Returns:
        pd.DataFrame: Корпус новостей с добавленным столбцом тональности.
    """
    corpus['sentiment'] = get_sentiment(corpus['title'])
    return corpus

def run_aggregator():
    SessionLocal = sessionmaker(bind=engine)
    db: Session = SessionLocal()
    try:
        corpus = get_corpus(db)
        corpus_df = prepare_corpus(corpus)
        corpus_with_sentiment = add_sentiment_to_corpus(corpus_df)
        save_sentiment_to_db(db, corpus_with_sentiment)
    finally:
        db.close()
    return corpus_with_sentiment

if __name__ == "__main__":
    print(run_aggregator())
