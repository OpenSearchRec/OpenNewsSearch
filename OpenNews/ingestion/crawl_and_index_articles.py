from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer
import time

from OpenSearchRec.retrieval import (
    IndexConfig,
    EmbeddingConfig,
    EmbeddingComparisonMetric,
    SearchItem,
    SearchConfig
)


from OpenNews.utils.news_utils import (
    get_normalized_url,
    get_domain_from_url,
    get_article_dict
)


def check_if_news_article_index_exists(opensearchrec_client, index_name):
    return index_name in opensearchrec_client.list_all_indexes()


def delete_news_article_index(opensearchrec_client, index_name):
    return opensearchrec_client.delete_index(index_name)


def create_news_article_index(opensearchrec_client, index_name):
    index_config = IndexConfig(
        text_matching_config={
            "text_matching_type": "bm25",
            "settings": {
                "bm25_k1": 0.25,  # small boost for when a term matches multiple time in text
                "bm25_b": 0.0,  # no penalization of longer text
                "enable_ngram_tokenizer": True,
                "analyzer": "english",
                "ngram_tokenizer_min_gram": 2,
                "ngram_tokenizer_max_gram": 4
            }
        },
        text_fields=[
            "article_title",
            "article_text",
            "article_text_beginning",
            "article_entities",
            "article_source_name",
            "article_authors"
        ],
        categorical_fields=[
            "source_name",
            "article_normalized_url",
            "article_normalized_title",
            "article_language_code",
            "article_news_category_tags",
            "article_normalized_domain_name",
            "article_authors",
            "article_centroid_tags",
            "similar_article_ids"
        ],
        date_fields=[
            "publish_date",
            "first_indexed_date",
            "last_update_in_index_date"
        ],
        numeric_fields=[
            "centrality_score_3",
            "centrality_score_10",
            "centrality_score_15",
            "centrality_score_half",
            "centrality_score_all"
        ],
        embedding_fields={
            "title_embedding": EmbeddingConfig(
                embedding_dimension=768,
                enable_approximate_nearest_embedding_search=True,
                approximate_nearest_embedding_search_metric=EmbeddingComparisonMetric.cosine_similarity
            )
        }
    )
    return  opensearchrec_client.create_index(index_name, index_config)  # throws error if index exists already


def index_setup_logic(
        opensearchrec_client, index_name,
        create_news_index_if_not_already_exists=True,
        delete_and_recreate_news_index_if_already_exists=False,
        logger=False):
    index_already_exists = check_if_news_article_index_exists(opensearchrec_client, index_name)
    if logger: logger.info(f"index_already_exists {index_already_exists}")
    if not index_already_exists:
        if not create_news_index_if_not_already_exists:
            raise Exception("Index is does not already exists and create_news_index_if_not_already_exists=False")
        else:
            if logger: logger.info(f"creating index {index_name}")
            create_news_article_index(opensearchrec_client, index_name)
    else:  # index exists
        if delete_and_recreate_news_index_if_already_exists:
            if logger: logger.info(f"deleting index {index_name}")
            delete_news_article_index(opensearchrec_client, index_name)
            if logger: logger.info(f"creating index {index_name}")
            create_news_article_index(opensearchrec_client, index_name)



def get_article_id_from_article_url(url):
    return url.lower().replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_")


def get_article_id_if_duplicate(opensearchrec_client, index_name, article_dict):
    """
        Article is a duplicate if normalized url already there, in  article_normalized_url field
    """
    search_response = \
        opensearchrec_client.search(
            index_name, 
            SearchConfig.parse_obj({
                "categorical_matching": [{
                    "categorical_field_name": "article_normalized_url",
                    "values_list": [article_dict["normalized_url"]],
                    "score_multiplier": 1,
                    "minimum_should_match": 1,
                    "required": True
                }],
                "limit": 1,
                "return_item_data": False
            }),
            
        )
    if len(search_response.results) > 0:
        return search_response.results[0].id
    return None


def index_article(opensearchrec_client, index_name, article_dict, embedding_model, source_name=None, logger=False):
    duplicate_article_id = get_article_id_if_duplicate(opensearchrec_client, index_name, article_dict)

    if duplicate_article_id is None:
        article = \
            SearchItem(
                # id=get_article_id_from_article_url(article_dict["normalized_url"]),
                text_fields={
                    "article_title": article_dict["title"],
                    "article_text": article_dict["text"],
                    "article_text_beginning": " ".join(article_dict["text"].split(" ")[:100]), # first ~100 words
                    "article_authors": ", ".join(article_dict["authors"]),
                    # "article_source_name": source_name
                },
                date_fields={
                    "publish_date": article_dict["publish_date"],
                    "first_indexed_date": datetime.utcnow()
                },
                categorical_fields={
                    # "source_name": source_name,
                    "article_authors": article_dict["authors"],
                    "article_normalized_url": article_dict["normalized_url"],
                    "article_normalized_domain_name": article_dict["normalized_domain_name"],
                    "article_normalized_title": article_dict["title"],
                    "article_language_code": article_dict["language"],
                    # "article_centroid_tags",  # not available yet
                    # "similar_article_ids" # not available yet
                },
                embedding_fields={
                    "title_embedding": embedding_model.encode(article_dict["title"]).tolist(),
                },
                extra_information={
                    "top_image": article_dict["top_image"],
                    "movies": article_dict["movies"],
                    "article_url": article_dict["url"]
                }
            )

        if source_name is not None:
            article.categorical_fields["source_name"] = source_name
            article.text_fields["article_source_name"] = source_name

        r = opensearchrec_client.index_item(index_name, article, refresh="wait_for")
        if logger: logger.info(r)
    else:
        if logger: logger.info(f"duplicate article normalized_url = {article_dict['normalized_url']}")

    return


def crawl_and_index_articles(
        article_url_list, opensearchrec_client, index_name,
        allowed_language_codes=["en"],
        create_news_index_if_not_already_exists=True,
        delete_and_recreate_news_index_if_already_exists=False,
        embedding_model_path="all-mpnet-base-v2",
        source_name=None,
        logger=False,
        sleep_seconds_between_articles=0,
        run_index_setup_logic=True):

    if run_index_setup_logic:
        index_setup_logic(
            opensearchrec_client, index_name,
            create_news_index_if_not_already_exists=create_news_index_if_not_already_exists,
            delete_and_recreate_news_index_if_already_exists=delete_and_recreate_news_index_if_already_exists,
            logger=logger)

        if logger: logger.info(f"opensearchrec_client.list_all_indexes() = {opensearchrec_client.list_all_indexes()}")


    embedding_model = SentenceTransformer(embedding_model_path)

    for a_idx, article_url in enumerate(article_url_list):
        if logger: logger.info(f"{a_idx} / {len(article_url_list)}")
        if not article_url.startswith("http://") and not article_url.startswith("https://"):
            article_url = "https://" + article_url
        
        try:
            article_dict = get_article_dict(article_url)
        
            if article_dict is not None:
                if logger: logger.debug(f"article language {article_dict['language']}")
                if article_dict["language"] in allowed_language_codes:
                    if logger: logger.info(f"indexing {article_url}, language = {article_dict['language']}")
                    if logger: logger.debug(article_dict)
                    index_article(opensearchrec_client, index_name, article_dict, embedding_model, source_name=source_name, logger=logger)
                else:
                    if logger: logger.info(f"skipping {article_url}, language = {article_dict['language']}")

        except Exception as e:
            if logger: logger.exception("\n\n" + str(e))
            if logger: logger.info(article_url + "\n\n")
            article_dict = None
        
        
        time.sleep(sleep_seconds_between_articles)

    return
