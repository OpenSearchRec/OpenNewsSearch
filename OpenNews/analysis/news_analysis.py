import argparse
from datetime import datetime, timedelta
import logging
import os
import yaml

import numpy as np
from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from OpenSearchRec.retrieval import (
    ElasticSearchRetrievalClientSettings,
    ElasticSearchRetrievalClient,
    SearchConfig,
    SearchItem
)

from OpenSearchRec.post_ranking import (
    select_items_greedily_and_with_similarity_suppression,
    select_top_k_centroids
)


from OpenNews.utils.entity_extraction import get_entity_token_list

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


def compute_articles_tfidf_matrix(articles_list, article_texts):


    # article_titles = [a.item["text_fields"]["article_title"] for a in articles_list]
    # article_texts = [a.item["text_fields"]["article_text"] for a in articles_list]

    # print("article_texts", article_texts)
    text_parsing_args = {
        "strip_accents": "unicode",
        "lowercase": True,
        "ngram_range": (1, 1)
    }

    if not use_generic_news_data_for_idf:
        tfidf_vectorizer = TfidfVectorizer(
            **text_parsing_args,
            smooth_idf=True,
            analyzer="word",
        )
        return tfidf_vectorizer.fit_transform(article_texts)

    count_vect = CountVectorizer(
        **text_parsing_args
    )
    count_vect.fit(article_texts)  # use the actual articles to determine the vocabulary
    tfidf_vectorizer = TfidfVectorizer(
        **text_parsing_args,
        smooth_idf=True,
        analyzer="word",
    )
    tfidf_vectorizer.fit(generic_news_data) # use the generic news data to determine idf term
    return tfidf_vectorizer.transform(article_texts)

def compute_article_similarity_matrix(
        articles_list,
        features_matrix,
        similarity_multiplier_for_different_articles_from_same_source=0,
        min_similarity_percentile_for_non_zero=75):

    similarities = linear_kernel(features_matrix, features_matrix)

    if similarity_multiplier_for_different_articles_from_same_source is not None:
        article_sources = [
            a.item["categorical_fields"]["article_normalized_domain_name"]
            for a in articles_list
        ]
        for i in range(len(article_sources)):
            for j in range(i + 1, len(article_sources)):
                if (i != j) and (article_sources[i] == article_sources[j]):
                    similarities[i, j] *= similarity_multiplier_for_different_articles_from_same_source
                    similarities[j, i] *= similarity_multiplier_for_different_articles_from_same_source

    if min_similarity_percentile_for_non_zero is not None:
        non_diagonal_similarities = np.copy(similarities)
        non_diagonal_similarities[np.diag_indices_from(non_diagonal_similarities)] = 0
        min_similarity_value = np.percentile(non_diagonal_similarities, [min_similarity_percentile_for_non_zero])
        similarities[similarities < min_similarity_value] = 0

    similarities[similarities < 0] = 0
    similarities[similarities > 1] = 1

    return similarities


def compute_centrality_scores(
        articles_list,
        article_similarity_matrix):

    article_sources = np.array([
        a.item["categorical_fields"]["article_normalized_domain_name"]
        for a in articles_list
    ])

    sources_list = list(set(article_sources))

    # for each article, compute the most similar for each other source
    article_to_most_similar_per_source_list = [[] for _ in articles_list]
    for a_idx, article in enumerate(articles_list):
        for source_name in sources_list:
            if article_sources[a_idx] != source_name:
                articles_with_source_idxes = article_sources==source_name
                most_similar_article_score = np.max(article_similarity_matrix[a_idx, articles_with_source_idxes])
                # print("most_similar_article_score", most_similar_article_score)
                article_to_most_similar_per_source_list[a_idx].append(most_similar_article_score)

    for a_idx in range(len(article_to_most_similar_per_source_list)):
        article_to_most_similar_per_source_list[a_idx].sort(reverse=True)
    
    centrality_score_3 = [
        np.mean(most_similar_per_source[:3]) for most_similar_per_source in article_to_most_similar_per_source_list
    ]
    centrality_score_10 = [
        np.mean(most_similar_per_source[:10]) for most_similar_per_source in article_to_most_similar_per_source_list
    ]
    centrality_score_15 = [
        np.mean(most_similar_per_source[:15]) for most_similar_per_source in article_to_most_similar_per_source_list
    ]

    half_num_sources = int(np.ceil(len(sources_list) / 2))
    centrality_score_half = [
        np.mean(most_similar_per_source[:half_num_sources]) for most_similar_per_source in article_to_most_similar_per_source_list
    ]

    centrality_score_all = [
        np.mean(most_similar_per_source) for most_similar_per_source in article_to_most_similar_per_source_list
    ]

    return {
        "centrality_score_3": centrality_score_3,
        "centrality_score_10": centrality_score_10,
        "centrality_score_15": centrality_score_15,
        "centrality_score_half": centrality_score_half,
        "centrality_score_all": centrality_score_all
    }


def compute_most_similar_articles(
        articles_list,
        article_similarity_matrix):

    article_sources = np.array([
        a.item["categorical_fields"]["article_normalized_domain_name"]
        for a in articles_list
    ])

    sources_list = list(set(article_sources))
    
    article_to_most_similar_per_source_idx_list = [[] for _ in articles_list]
    article_to_most_similar_per_source_score_list = [[] for _ in articles_list]
    for a_idx, article in enumerate(articles_list):
        for source_name in sources_list:
            # print("\n\nsource_name", source_name)
            if article_sources[a_idx] != source_name:
                articles_with_source_idxes = (article_sources==source_name)*1
                # print("articles_with_source_idxes", articles_with_source_idxes)
                similarity_scores_from_source = article_similarity_matrix[a_idx]*articles_with_source_idxes
                most_similar_article_score = np.max(similarity_scores_from_source)
                most_similar_article_idx = np.argmax(similarity_scores_from_source)

                if most_similar_article_score > 0:
                    # print("articles_with_source_idxes", articles_with_source_idxes)
                    # print("similarity_scores_from_source", similarity_scores_from_source)
                    # print("most_similar_article_idx", most_similar_article_idx)
                    # print("most_similar_article_score", most_similar_article_score)

                    assert most_similar_article_idx not in article_to_most_similar_per_source_idx_list[a_idx]

                    article_to_most_similar_per_source_idx_list[a_idx].append(most_similar_article_idx)
                    article_to_most_similar_per_source_score_list[a_idx].append(most_similar_article_score)

    article_to_most_similar_per_source_title_list = []
    article_to_most_similar_per_source_url_list = []
    for most_similar_idxes in article_to_most_similar_per_source_idx_list:
        titles = []
        for idx in most_similar_idxes:
            titles.append(articles_list[idx].item["text_fields"]["article_title"])
        article_to_most_similar_per_source_title_list.append(titles)
        
        urls = []
        for idx in most_similar_idxes:
            urls.append(articles_list[idx].item["extra_information"]["article_url"])
        article_to_most_similar_per_source_url_list.append(urls)

    similar_articles = []
    for a_idx in range(len(article_to_most_similar_per_source_score_list)):
        scores = article_to_most_similar_per_source_score_list[a_idx]
        idxes = article_to_most_similar_per_source_idx_list[a_idx]
        titles = article_to_most_similar_per_source_title_list[a_idx]
        urls = article_to_most_similar_per_source_url_list[a_idx]
        combined = list(zip(scores, idxes, titles, urls))
        # print("combined", combined)
        combined.sort(reverse=True)
        # print("combined", combined)
        # print("list(zip(*combined))", list(zip(*combined)))
        similar_articles.append([
            {
                "similarity": float(a[0]),
                "id": articles_list[int(a[1])].id,
                "title": a[2],
                "url": a[3]
            } for a in combined
        ])

        if len(article_to_most_similar_per_source_score_list[a_idx]) > 0:
            scores, idxes, titles, urls = list(zip(*combined))
            article_to_most_similar_per_source_score_list[a_idx] = scores
            article_to_most_similar_per_source_idx_list[a_idx] = idxes
            article_to_most_similar_per_source_title_list[a_idx] = titles
            article_to_most_similar_per_source_url_list[a_idx] = urls



    return {
        "article_to_most_similar_per_source_idx_list": article_to_most_similar_per_source_idx_list,
        "article_to_most_similar_per_source_title_list": article_to_most_similar_per_source_title_list,
        "article_to_most_similar_per_source_url_list": article_to_most_similar_per_source_url_list,
        "article_to_most_similar_per_source_score_list": article_to_most_similar_per_source_score_list,
        "similar_articles": similar_articles
    }


def compute_centroids_tags(
        articles_list,
        articles_scores,
        article_similarity_matrix,
        k_list=[5, 10, 20, 50]):

    centroid_tags = [[] for _ in articles_list]

    for k in k_list:
        if len(articles_list) > k * 5:
            centroids_results = \
                select_top_k_centroids(
                    num_centroids=k,
                    items_similarity_matrix=np.array(article_similarity_matrix),
                    items_weights=np.array(articles_scores),
                    run_input_validation=False)
            centroid_idx_list = centroids_results["sorted_centroid_idx_list"]
            # centroid_scores_list = centroids_results["sorted_centroid_scores_list"]
            for centroid_idx in centroid_idx_list:
                centroid_tags[centroid_idx].append(f"centroid_k{k}")

    return centroid_tags



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

        article_title_and_text_beginning_tfidf_matrix = compute_articles_tfidf_matrix(articles_list, article_title_and_text_beginning)
        article_title_entities_tfidf_matrix = compute_articles_tfidf_matrix(articles_list, article_title_entities)
        article_text_entities_tfidf_matrix = compute_articles_tfidf_matrix(articles_list, article_text_entities)
        
        article_similarity_matrix = \
            compute_article_similarity_matrix(
                articles_list,
                article_title_and_text_beginning_tfidf_matrix,
                similarity_multiplier_for_different_articles_from_same_source=0,
                min_similarity_percentile_for_non_zero=95)
        
        article_title_entity_similarity_matrix = \
            compute_article_similarity_matrix(
                articles_list,
                article_title_entities_tfidf_matrix,
                similarity_multiplier_for_different_articles_from_same_source=0,
                min_similarity_percentile_for_non_zero=95)

        article_text_entity_similarity_matrix = \
            compute_article_similarity_matrix(
                articles_list,
                article_text_entities_tfidf_matrix,
                similarity_multiplier_for_different_articles_from_same_source=0,
                min_similarity_percentile_for_non_zero=95)

        # article_similarity_matrix_strict = \
        #     compute_article_similarity_matrix(
        #         articles_list,
        #         article_title_and_text_beginning_tfidf_matrix,
        #         similarity_multiplier_for_different_articles_from_same_source=0,
        #         min_similarity_percentile_for_non_zero=98)

        combined_article_similarity_matrix = (article_similarity_matrix + article_title_entity_similarity_matrix + article_text_entity_similarity_matrix) / (3.0 + 1e-12)
        
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
