import pdb
import time
import traceback
from datetime import datetime, timedelta
from typing import List
from uuid import uuid4

from loguru import logger

from service_aggregator.embedder import get_embeddings_text
from service_repository.dependencies import SessionManager
from service_repository.models import NewsModel, NewsEmbModel
from service_repository.settings import APP_SETTINGS
from service_scraper.spiders import (
    RBCParser, RIAParser
)

SOURCES = {
    "РБК": [
        RBCParser("https://quote.rbc.ru/?utm_source=topline")
    ],
    "РИА НОВОСТИ": [
        RIAParser("https://ria.ru/economy/")
    ]
}


def get_last_post_dttm(period_days: int) -> datetime:
    return (datetime.now() - timedelta(days=period_days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def save_last_news(news: List[dict], last_post_dttm: datetime):
    news_models = []
    news_emd_models = []
    processed_dttm = datetime.now().replace(tzinfo=None)

    with SessionManager() as db:
        urls = (
            db.query(NewsModel.url)
            .filter(NewsModel.post_dttm >= (last_post_dttm - timedelta(days=10)))
            .all()
        )
        urls = {url[0] for url in urls}

    for row in news:
        if row["url"] in urls:
            continue
        news_models.append(
            NewsModel(
                uuid=row["uuid"],
                title=row["title"],
                post_dttm=row["post_dttm"],
                url=row["url"],
                processed_dttm=processed_dttm,
                polarity=row.get("polarity")
            )
        )
        #news_emd_models.append(
            #NewsEmbModel(
                #uuid=row["uuid"],
                #embedding_full_text=row["embedding_full_text"],
                #embedding_title=row["embedding_title"],
           # )
       # )
    with SessionManager() as db:
        if news_models:
            logger.info("put NewsModel")
            db.bulk_save_objects(news_models)
            logger.info("put NewsEmbModel")
            db.bulk_save_objects(news_emd_models)
            db.commit()
        else:
            logger.info("Skip DB")

def preproccess_news(news: List[dict]):
    """Определяет является новость по title и строит эмбединги"""
    news_filtered = []
    for row in news:
        row["uuid"] = uuid4()
        row["embedding_full_text"] = get_embeddings_text([row["full_text"]])[0]
        row["embedding_title"] = get_embeddings_text([row["title"]])[0]
        news_filtered.append(row)
    logger.info(f"Raw news {len(news)}, filetred {len(news_filtered)}")
    return news

def run_spider():
    while True:
        last_post_dttm = get_last_post_dttm(APP_SETTINGS.SPIDER_PERIOD_DAYS)
        logger.info(f"last_post_dttm {last_post_dttm}")
        for source in SOURCES:
            for parser in SOURCES[source]:
                logger.info(f"Start {source} {parser.pages_url}")
                try:
                    news = parser.parse(stop_datetime=last_post_dttm)
                    if news:
                        #news = preproccess_news(news)
                        save_last_news(news, last_post_dttm)
                        logger.info(f"Done {source} {parser.pages_url}")
                    else:
                        logger.error(f"No news for {source} {parser.pages_url}")
                except Exception as err:
                    trace = traceback.format_exc()
                    logger.error(f"Unexpected exception {err} trace {trace}")
        logger.info("sleep")
        time.sleep(APP_SETTINGS.SPIDER_WAIT_TIMEOUT_SEC)


if __name__ == "__main__":
    run_spider()

