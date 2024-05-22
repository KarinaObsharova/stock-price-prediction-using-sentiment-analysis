from sqlalchemy import ARRAY, TIMESTAMP, Column, Float, String
from sqlalchemy.ext.mutable import MutableList

from service_repository.database import Base


class NewsModel(Base):
    """Таблица новостей"""

    __tablename__ = "news"

    uuid = Column(String, primary_key=True)
    title = Column(String)
    post_dttm = Column(TIMESTAMP)
    url = Column(String, unique=True)
    processed_dttm = Column(TIMESTAMP)
    label = Column(String)


class NewsEmbModel(Base):
    """Таблица эмбедингов новостей"""

    __tablename__ = "news_emb"

    uuid = Column(String, primary_key=True)
    embedding_full_text = Column(MutableList.as_mutable(ARRAY(Float)))
    embedding_title = Column(MutableList.as_mutable(ARRAY(Float)))
