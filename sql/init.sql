create table news(
    uuid text,
    title text NOT NULL,
    post_dttm timestamp,
    url text NOT NULL UNIQUE,
    processed_dttm timestamp,
    PRIMARY KEY (uuid)
);
CREATE INDEX ix_news_processed_dttm ON news (processed_dttm);

create table news_emb(
    uuid text,
    embedding_full_text real[] NOT NULL,
    embedding_title real[] NOT NULL,
    PRIMARY KEY (uuid)
);

ALTER TABLE news
ADD label text;