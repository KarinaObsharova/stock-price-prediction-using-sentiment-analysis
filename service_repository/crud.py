from sqlalchemy.orm import sessionmaker
from service_repository.models import NewsModel, NewsEmbModel
from typing import List, Tuple

from sqlalchemy import and_

# service_repository/crud.py

from sqlalchemy.orm import Session
from service_repository.models import NewsModel

def get_corpus(db: Session):
    """Получение корпуса новостей из БД"""
    corpus = db.query(NewsModel).all()
    return corpus

