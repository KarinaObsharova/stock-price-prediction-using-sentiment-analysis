from typing import List
from sklearn.cluster import AgglomerativeClustering

CLUSTER_N_CLUSTER = 3
CLUSTER_LINKAGE = "average"


def get_clusters(corpus_embeddings: List[str]) -> List[int]:
    """Кластеризация корпуса текстов по содержанию.

    Args:
        corpus_embeddings (List[str]): список эмбедингов текстов

    Returns:
        List[int]: список кластеров. i - ый текст -> a[i] кластер.
    """

    clustering = AgglomerativeClustering(
        n_clusters=CLUSTER_N_CLUSTER,
        linkage=CLUSTER_LINKAGE,
    )

    clustering.fit(corpus_embeddings)

    return clustering.labels_
