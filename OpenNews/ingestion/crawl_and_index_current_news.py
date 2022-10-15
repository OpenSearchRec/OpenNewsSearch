import argparse
import pandas as pd
import logging
import os
import yaml
import time

from OpenSearchRec.retrieval import (
    ElasticSearchRetrievalClientSettings,
    ElasticSearchRetrievalClient
)


from OpenNews.utils.news_utils import (
    get_normalized_url,
    get_domain_from_url,
    get_page_url_from_domain
)


from OpenNews.ingestion.crawl_and_index_articles import (
    crawl_and_index_articles
)


if __name__ == "__main__":
    logger = logging.getLogger('current_news_crawler_indexer')
    logger.setLevel(logging.INFO)
    logging_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging_formatter)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_filepath', type=str, default="configs/open_news_config.yml")
    parser.add_argument('--repeat_crawl_in_infinite_loop', default=False, action='store_true')

    parser.add_argument('--opensearchrec_database_type',  type=str, default=os.getenv("opensearchrec_database_type", "elasticsearch"))
    parser.add_argument('--opensearchrec_database_url',  type=str, default=os.getenv("opensearchrec_database_url", "http://localhost:9200"))
    parser.add_argument('--opensearchrec_database_username',  type=str, default=os.getenv("opensearchrec_database_username", "elastic"))
    parser.add_argument('--opensearchrec_database_password',  type=str, default=os.getenv("opensearchrec_database_password", "admin"))
    parser.add_argument('--elasticsearch_verify_certificates',  type=str, default=os.getenv("elasticsearch_verify_certificates", "False"))

    parser.add_argument('--create_news_index_if_not_already_exists',  type=bool, default=True)
    parser.add_argument('--delete_and_recreate_news_index_if_already_exists',  type=bool, default=False)
    parser.add_argument('--embedding_model_path',  type=str, default="all-mpnet-base-v2")

    args = parser.parse_args()
    logger.info(args)

    with open(args.config_filepath, "r") as f:
        config = yaml.safe_load(f)
    logger.info(config)
    

    news_index_name = config["news_index_name"]

    allowed_language_codes = config["allowed_language_codes"]

    logger.info(f"news_index_name = {news_index_name}")


    open_search_rec_settings = \
        ElasticSearchRetrievalClientSettings(
            database_type=args.opensearchrec_database_type,
            elasticsearch_host=args.opensearchrec_database_url,
            elasticsearch_index_prefix="opensearchrec_index_prefix_",
            elasticsearch_alias_prefix="opensearchrec_alias_prefix_",
            elasticsearch_username=args.opensearchrec_database_username,
            elasticsearch_password=args.opensearchrec_database_password,
            elasticsearch_verify_certificates=args.elasticsearch_verify_certificates
        )

    logger.info(f"open_search_rec_settings = {open_search_rec_settings}")

    opensearchrec_client = ElasticSearchRetrievalClient(open_search_rec_settings)


    iteration_number = 1
    while args.repeat_crawl_in_infinite_loop or iteration_number == 1:
        logger.info(f"iteration_number = {iteration_number}")

        for source_name, source_info in config["news_sources"].items():
            article_url_list = get_page_url_from_domain(source_info["domain"], memoize_articles=False)
            print(source_name)
            print(article_url_list)

            crawl_and_index_articles(
                article_url_list, opensearchrec_client, news_index_name,
                allowed_language_codes=allowed_language_codes,
                embedding_model_path=args.embedding_model_path,
                source_name=source_name,
                create_news_index_if_not_already_exists=args.create_news_index_if_not_already_exists,
                delete_and_recreate_news_index_if_already_exists=args.delete_and_recreate_news_index_if_already_exists,
                logger=logger,
                sleep_seconds_between_articles=0)

            
        iteration_number += 1
