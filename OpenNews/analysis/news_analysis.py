import argparse
from datetime import datetime, timedelta
import logging
import os
import yaml

import numpy as np
from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from sentence_transformers import SentenceTransformer

from OpenSearchRec.retrieval import (
    ElasticSearchRetrievalClientSettings,
    ElasticSearchRetrievalClient,
    SearchConfig,
    SearchItem
)


from OpenNews.utils.entity_extraction import get_entity_token_list


from OpenNews.analysis.news_analysis_utils import *


embedding_model = SentenceTransformer("all-mpnet-base-v2")


use_generic_news_data_for_idf = True

if use_generic_news_data_for_idf:
    from sklearn.datasets import fetch_20newsgroups
    news_data, _ = fetch_20newsgroups(return_X_y=True)
    generic_news_data = list(news_data)

def get_all_articles_in_time_interval(
        opensearchrec_client,
        news_index_name,
        min_article_datetime_for_analysis,
        max_article_datetime_for_analysis):
    """
        TODO: Need to improve pagination functionality in OpenSearchRec, such as to allow sort by _id
    """
    page_size_limit = 10000
    article_list = []
    article_id_set = set() 
    while True:
        search_response = opensearchrec_client.search(news_index_name, SearchConfig.parse_obj({
            "date_fields_filtering": [
                {
                    "input_value_config": {
                        "field_name": "publish_date"
                    },
                    "minimum_value": min_article_datetime_for_analysis,
                    "maximum_value": max_article_datetime_for_analysis
                }
            ],
            "date_fields_boosting": [ # date boosting for pagination purposes
                {
                    "input_value_config": {
                        "field_name": "publish_date",
                        "target_date": str(datetime.utcnow()),
                        "time_units": "hours"
                    },
                    "boost_function_config": {
                        "decay_boost_type": "exponential",
                        "decay_boost_offset": 24,
                        "decay_boost_decay_rate": 0.5,
                        "decay_boost_decay_scale": 12
                    },
                    "scoring_factor_config": {
                        "score_multiplier_constant_weight": 0,
                        "score_multiplier_variable_boost_weight": 1,
                        "score_multiplier_minimum_value": 0,
                        "score_multiplier_maximum_value": 2
                    }
                }
            ],
            "start": len(article_list),
            "limit": page_size_limit
        }))

        for result in search_response.results:
            if result.id not in article_id_set:
                article_id_set.add(result.id)
                article_list.append(result)

        if len(search_response.results) < page_size_limit:
            break
    
    return article_list





def compute_composite_compute_article_similarity_matrix(
        articles_list,
        article_title_entities,
        article_text_entities,
        similarity_multiplier_for_different_articles_from_same_source=0,
        min_similarity_percentile_for_non_zero=75,
        logger=False):


    article_title_and_text_beginning = [
        " ".join([a.item["text_fields"]["article_title"]]) + a.item["text_fields"]["article_text_beginning"]
        for a in articles_list
    ]
    article_titles = [a.item["text_fields"]["article_title"] for a in articles_list]
    article_texts = [a.item["text_fields"]["article_text"] for a in articles_list]
    

    # TF-IDF
    article_title_and_text_beginning_tfidf_matrix = compute_articles_tfidf_matrix(articles_list, article_title_and_text_beginning)
    article_similarity_matrix = \
        compute_article_similarity_matrix(
            articles_list,
            article_title_and_text_beginning_tfidf_matrix,
            similarity_multiplier_for_different_articles_from_same_source=0,
            min_similarity_percentile_for_non_zero=min_similarity_percentile_for_non_zero)


    # Entity Extraction
    article_title_entities_tfidf_matrix = compute_articles_tfidf_matrix(articles_list, article_title_entities)
    article_text_entities_tfidf_matrix = compute_articles_tfidf_matrix(articles_list, article_text_entities)
    

    article_title_entity_similarity_matrix = \
        compute_article_similarity_matrix(
            articles_list,
            article_title_entities_tfidf_matrix,
            similarity_multiplier_for_different_articles_from_same_source=0,
            min_similarity_percentile_for_non_zero=min_similarity_percentile_for_non_zero)

    article_text_entity_similarity_matrix = \
        compute_article_similarity_matrix(
            articles_list,
            article_text_entities_tfidf_matrix,
            similarity_multiplier_for_different_articles_from_same_source=0,
            min_similarity_percentile_for_non_zero=min_similarity_percentile_for_non_zero)

    entity_similarity_matrix = (article_title_entity_similarity_matrix + article_text_entity_similarity_matrix) / (2 + 1e-12)


    # Semantic Embeddings
    article_title_semantic_embeddings = [embedding_model.encode(title).tolist() for title in article_titles]
    article_sementic_embedding_similarity_matrix = \
        compute_article_similarity_matrix(
            articles_list,
            article_title_semantic_embeddings,
            similarity_multiplier_for_different_articles_from_same_source=0,
            min_similarity_percentile_for_non_zero=min_similarity_percentile_for_non_zero)

    # Combine Similarity Matrices
    tf_idf_weight = 1
    entity_weight = 1
    title_semantic_embedding_weight = 1
    combined_article_similarity_matrix = (
        article_similarity_matrix * tf_idf_weight + \
        entity_similarity_matrix * entity_weight + \
        article_sementic_embedding_similarity_matrix * title_semantic_embedding_weight) / (
            tf_idf_weight + entity_weight + title_semantic_embedding_weight + 1e-12)

    return combined_article_similarity_matrix



def perform_news_analysis(
        opensearchrec_client,
        news_index_name,
        min_article_datetime_for_analysis,
        max_article_datetime_for_analysis,
        num_days_for_expanded_articles_list=1,
        logger=False):

    articles_list = \
        get_all_articles_in_time_interval(
            opensearchrec_client,
            news_index_name,
            min_article_datetime_for_analysis,
            max_article_datetime_for_analysis)

    expanded_articles_list = \
        get_all_articles_in_time_interval(
            opensearchrec_client,
            news_index_name,
            min_article_datetime_for_analysis - timedelta(days=num_days_for_expanded_articles_list),
            max_article_datetime_for_analysis + timedelta(days=num_days_for_expanded_articles_list))

    if logger: logger.info(f"num articles = {len(articles_list)}")

    article_sources = [
        a.item["categorical_fields"]["article_normalized_domain_name"]
        for a in articles_list
    ]

    sources_list = list(set(article_sources))


    if len(articles_list) > 10 and len(sources_list) > 1:
        
        article_title_and_text_beginning = [
            " ".join([a.item["text_fields"]["article_title"]]) + a.item["text_fields"]["article_text_beginning"]
            for a in articles_list
        ]
        article_titles = [a.item["text_fields"]["article_title"] for a in articles_list]
        article_texts = [a.item["text_fields"]["article_text"] for a in articles_list]
        
        if logger: logger.info("start extracting entities")
        article_title_entities = []
        article_text_entities = []
        for idx, (title, text) in enumerate(list(zip(article_titles, article_texts))):
            if logger and (idx%100==0): logger.info(f"entity extraction: {idx}/{len(article_titles)}")
            article_title_entities.append(" ".join(get_entity_token_list(title)))
            article_text_entities.append(" ".join(get_entity_token_list(" ".join(text.split(" ")[:300]) )))
            # print(title)
            # print(article_title_entities[-1])
        if logger: logger.info("done extracting entities")

        combined_article_similarity_matrix = \
            compute_composite_compute_article_similarity_matrix(
                articles_list,
                article_title_entities,
                article_text_entities,
                similarity_multiplier_for_different_articles_from_same_source=0,
                min_similarity_percentile_for_non_zero=90,
                logger=logger)

        # print("combined_article_similarity_matrix.min()", combined_article_similarity_matrix.min(), flush=True)
        # print("combined_article_similarity_matrix.min() >= 0", combined_article_similarity_matrix.min() >= 0, flush=True)
        
        centrality_scores = compute_centrality_scores(articles_list, combined_article_similarity_matrix)


        # print("centrality_score_half", centrality_scores["centrality_score_half"], flush=True)
        # print("min centrality_score_half >= 0", min(centrality_scores["centrality_score_half"]) >= 0, flush=True)

        # print("centrality_scores", centrality_scores)

        # top_article_idx = np.argmax(centrality_scores["centrality_score_10"])
        # print(articles_list[top_article_idx].item["text_fields"]["article_title"])
        # print(np.max(centrality_scores["centrality_score_10"]))

        # min_article_idx = np.argmin(centrality_scores["centrality_score_10"])
        # print(articles_list[min_article_idx].item["text_fields"]["article_title"])
        # print(np.min(centrality_scores["centrality_score_10"]))


        similar_articles = compute_most_similar_articles(articles_list, combined_article_similarity_matrix)
        # for i in range(min(30, len(articles_list))):
        #     print(articles_list[i].item["text_fields"]["article_title"])
        #     print("\t", similar_articles["similar_articles"][i])
        #     print("\t", similar_articles["article_to_most_similar_per_source_idx_list"][i])
        #     print("\t", similar_articles["article_to_most_similar_per_source_title_list"][i])
        #     print("\t", similar_articles["article_to_most_similar_per_source_score_list"][i])
        # print("articles_list[0]", articles_list[0])


        centroid_tags = \
            compute_centroids_tags(
                articles_list, centrality_scores["centrality_score_half"], combined_article_similarity_matrix)

        # if logger: logger.info(f"centroid_tags {centroid_tags}")

        article_update_list = []
        for a_idx, article in enumerate(articles_list):
            item_dict = {
                "id": article.id,
                "text_fields": {
                    "article_entities": article_title_entities[a_idx] + article_text_entities[a_idx]
                },
                "categorical_fields": {
                    "article_centroid_tags": centroid_tags[a_idx]
                },
                "numeric_fields": {
                    "centrality_score_3": centrality_scores["centrality_score_3"][a_idx],
                    "centrality_score_10": centrality_scores["centrality_score_10"][a_idx],
                    "centrality_score_15": centrality_scores["centrality_score_15"][a_idx],
                    "centrality_score_half": centrality_scores["centrality_score_half"][a_idx],
                    "centrality_score_all":  centrality_scores["centrality_score_all"][a_idx]
                },
            }

            item_dict["extra_information"] = article.item["extra_information"]
            item_dict["extra_information"]["similar_articles"] = similar_articles["similar_articles"][a_idx]
            
            article_update_list.append(SearchItem.parse_obj(item_dict))

        # print("article_update_list[0]", article_update_list[0])

        opensearchrec_client.bulk_update_items(news_index_name, article_update_list)




if __name__ == "__main__":
    logger = logging.getLogger('news_analysis')
    logger.setLevel(logging.INFO)
    logging_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging_formatter)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_filepath', type=str, default="configs/open_news_config.yml")
    parser.add_argument('--min_num_days_back', type=int, default=0)
    parser.add_argument('--max_num_days_back', type=int, default=3)
    parser.add_argument('--repeat_analysis_in_infinite_loop', action='store_true', default=False)

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
    max_days_difference_for_non_zero_article_similarity = float(config["news_analysis"]["max_days_difference_for_non_zero_article_similarity"])

    min_num_days_back = args.min_num_days_back
    max_num_days_back = args.max_num_days_back
    

    logger.info(f"news_index_name = {news_index_name}")
    logger.info(f"min_num_days_back = {min_num_days_back}")
    logger.info(f"max_num_days_back = {max_num_days_back}")
    logger.info(f"max_days_difference_for_non_zero_article_similarity = {max_days_difference_for_non_zero_article_similarity}")


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
    while args.repeat_analysis_in_infinite_loop or iteration_number == 1:
        logger.info(f"iteration_number = {iteration_number}")

        for num_days_back in range(min_num_days_back, max_num_days_back + 1):

            min_article_datetime_for_analysis = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=num_days_back)
            max_article_datetime_for_analysis = min_article_datetime_for_analysis + timedelta(days=1, microseconds=-1)

            logger.info(f"num_days_back = {num_days_back}")
            logger.info(f"min_article_datetime_for_analysis = {min_article_datetime_for_analysis}")
            logger.info(f"max_article_datetime_for_analysis = {max_article_datetime_for_analysis}")

            perform_news_analysis(
                opensearchrec_client, news_index_name,
                min_article_datetime_for_analysis=min_article_datetime_for_analysis,
                max_article_datetime_for_analysis=max_article_datetime_for_analysis,
                logger=logger)

            
        iteration_number += 1
