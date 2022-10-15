import argparse
import pandas as pd
import logging
import os
import multiprocessing
import numpy as np

from OpenSearchRec.retrieval import (
    ElasticSearchRetrievalClientSettings,
    ElasticSearchRetrievalClient
)


from OpenNews.utils.news_utils import (
    get_normalized_url,
    get_domain_from_url
)


from OpenNews.ingestion.crawl_and_index_articles import (
    crawl_and_index_articles,
    index_setup_logic
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--article_url_filepath',  type=str)
    parser.add_argument('--allowed_language_codes',  type=str, default="en",
                        help="Comma seperated list of ISO 639-1 language codes, with no spaces, for example --allowed_language_codes=en,fr")
    parser.add_argument('--opensearchrec_news_index_name',  type=str, required=True)

    parser.add_argument('--opensearchrec_database_type',  type=str, default=os.getenv("opensearchrec_database_type", "elasticsearch"))
    parser.add_argument('--opensearchrec_database_url',  type=str, default=os.getenv("opensearchrec_database_url", "http://localhost:9200"))
    parser.add_argument('--opensearchrec_database_username',  type=str, default=os.getenv("opensearchrec_database_username", "elastic"))
    parser.add_argument('--opensearchrec_database_password',  type=str, default=os.getenv("opensearchrec_database_password", "admin"))
    parser.add_argument('--elasticsearch_verify_certificates',  type=str, default=os.getenv("elasticsearch_verify_certificates", "False"))

    parser.add_argument('--create_news_index_if_not_already_exists',  type=bool, default=True)
    parser.add_argument('--delete_and_recreate_news_index_if_already_exists',  type=bool, default=False)
    parser.add_argument('--embedding_model_path',  type=str, default="all-mpnet-base-v2")
    parser.add_argument('--num_processes',  type=int, default=5)
    parser.add_argument('--random_index_order',  type=bool, default=True)

    args = parser.parse_args()
    print(args)

    logger = logging.getLogger('news_crawler_indexer')
    logger.setLevel(logging.INFO)
    logging_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging_formatter)
    logger.addHandler(console_handler)


    allowed_language_codes = args.allowed_language_codes.split(",")
    logger.info(f"allowed_language_codes {allowed_language_codes}")

    article_url_df = pd.read_csv(args.article_url_filepath)
    print(article_url_df[:10])

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

    opensearchrec_client = ElasticSearchRetrievalClient(open_search_rec_settings)

    article_url_list = list(article_url_df["url"])

    if args.random_index_order:
        np.random.shuffle(article_url_list)

    if args.num_processes == 1:
        crawl_and_index_articles(
            article_url_list, opensearchrec_client, args.opensearchrec_news_index_name,
            allowed_language_codes=allowed_language_codes,
            embedding_model_path=args.embedding_model_path,
            create_news_index_if_not_already_exists=args.create_news_index_if_not_already_exists,
            delete_and_recreate_news_index_if_already_exists=args.delete_and_recreate_news_index_if_already_exists,
            logger=logger,
            sleep_seconds_between_articles=0)
    else:
        url_sublists = np.array_split(article_url_list, args.num_processes)

        index_setup_logic(
                opensearchrec_client, args.opensearchrec_news_index_name,
                create_news_index_if_not_already_exists=args.create_news_index_if_not_already_exists,
                delete_and_recreate_news_index_if_already_exists=args.delete_and_recreate_news_index_if_already_exists,
                logger=logger)


        def process_url_list(url_list):
            crawl_and_index_articles(
                        url_list, opensearchrec_client, args.opensearchrec_news_index_name,
                        allowed_language_codes=allowed_language_codes,
                        embedding_model_path=args.embedding_model_path,
                        create_news_index_if_not_already_exists=False,
                        delete_and_recreate_news_index_if_already_exists=False,
                        run_index_setup_logic=False,
                        logger=logger,
                        sleep_seconds_between_articles=1)

  

        with multiprocessing.Pool(args.num_processes) as p:
            p.map(process_url_list, url_sublists)