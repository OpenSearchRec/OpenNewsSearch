from datetime import datetime, timedelta
from flask import Flask, request, render_template
import os
import yaml
import rapidfuzz

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from sentence_transformers import SentenceTransformer

from OpenSearchRec.retrieval import (
    ElasticSearchRetrievalClientSettings,
    ElasticSearchRetrievalClient,
    SearchConfig
)

from OpenSearchRec.post_ranking import (
    select_items_greedily_and_with_similarity_suppression,
    select_top_k_centroids,
    merge_lists
)

news_crawler_config_filepath = "configs/open_news_config.yml"
with open(news_crawler_config_filepath, "r") as f:
    config = yaml.safe_load(f)
index_name = config["news_index_name"]

embedding_model = SentenceTransformer("all-mpnet-base-v2")


code_url = os.getenv("code_url", "")
debugging = bool(os.getenv("debugging", False))
open_search_rec_settings = ElasticSearchRetrievalClientSettings()

print(f"open_search_rec_settings = {open_search_rec_settings}")

open_search_rec = ElasticSearchRetrievalClient(open_search_rec_settings)


app = Flask(__name__)


@app.route('/about')
def about():

    return render_template("about.html",
                           code_url=code_url,
                           configured_news_sources=config["news_sources"])




@app.route('/')
def home():
    request_json = {
        "date_fields_boosting": [
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
        "numeric_fields_boosting": [
            {
                "input_value_config": {
                    "field_name": "centrality_score_15",
                    "default_field_value_if_missing": 0
                },
                "boost_function_config": {
                    "boost_function_input_value_rescaling_factor": 1,
                    "boost_function_increasing_function_type": "none"
                },
                "scoring_factor_config": {
                    "score_multiplier_constant_weight": 0.1,
                    "score_multiplier_variable_boost_weight": 1,
                    "score_multiplier_minimum_value": 0,
                    "score_multiplier_maximum_value": 10
                }
            }
        ],
        "categorical_matching": [
            {
                "categorical_field_name": "article_centroid_tags",
                "values_list": [
                    # "centroid_k5", "centroid_k10", "centroid_k20", "centroid_k50", 
                    # "centroid_k20", 
                    "centroid_k50",
                ],
                "score_multiplier": 0,
                "minimum_should_match": 1,
                "required": True
            },
            {
                "categorical_field_name": "article_centroid_tags",
                "values_list": [
                    "centroid_k5", "centroid_k10", "centroid_k20",
                ],
                "score_multiplier": 2,
                "minimum_should_match": 1,
                "required": False
            },
            {
                "categorical_field_name": "article_centroid_tags",
                "values_list": [
                    "centroid_k5", "centroid_k10",
                ],
                "score_multiplier": 2,
                "minimum_should_match": 1,
                "required": False
            },
            {
                "categorical_field_name": "article_centroid_tags",
                "values_list": [
                    "centroid_k5",
                ],
                "score_multiplier": 2,
                "minimum_should_match": 1,
                "required": False
            }
        ],

        "embedding_fields_to_return": [],  # don't return embeddings to reduce data transfer and latency 
        "limit": 100,
        "start": 0
    }

    search_response = open_search_rec.search(index_name, SearchConfig.parse_obj(request_json))
    hits = search_response.results

    return render_template(
        "home.html", hits=hits,
        debugging=debugging
    )



def perform_search(
        open_search_rec, q, min_date, max_date, target_date_boost,
        require_all_terms_fuzzy_match=True,
        require_all_terms_exact_match=False,
        require_identified_as_centroid=False,
        require_one_term_exact_match=True,
        require_one_term_fuzzy_match=True,
        perform_fuzzy_matching_on_title_only=False,
        # use_rapidfuzz_for_filtering=False,
        limit=100):
    request_json = {
        "text_matching": [
            {
                "query": q,
                "text_fields_names_and_weights_dict": {
                    "article_title": 1.5,
                    "article_text_beginning": 0.1,
                    "article_text": 0.1,
                    "article_source_name": 1,
                    "article_authors": 1
                },
                "use_ngram_fields_if_available": True,
                "required": False,
                "minimum_should_match": 1
            },
            {
                "query": q,
                "text_fields_names_and_weights_dict": {
                    "article_title": 1.5,
                    "article_text_beginning": 0.1,
                    "article_source_name": 1,
                    "article_authors": 1
                },
                "use_ngram_fields_if_available": False,
                "required": False,
                "minimum_should_match": 1
            },
            {
                "query": q,
                "text_fields_names_and_weights_dict": {
                    "article_title": 1.5,
                    "article_text_beginning": 0.1,
                    "article_source_name": 1,
                    "article_authors": 1
                },
                "use_ngram_fields_if_available": False,
                "required": False,
                "minimum_should_match": "100%"
            }
        ],
        "date_fields_boosting": [
            {
                "input_value_config": {
                    "field_name": "publish_date",
                    "target_date": str(datetime.utcnow()),
                    "time_units": "hours"
                },
                "boost_function_config": {
                    "decay_boost_type": "exponential",
                    "decay_boost_offset": 48,
                    "decay_boost_decay_rate": 0.5,
                    "decay_boost_decay_scale": 12
                },
                "scoring_factor_config": {
                    "score_multiplier_constant_weight": 2,
                    "score_multiplier_variable_boost_weight": 1,
                    "score_multiplier_minimum_value": 0,
                    "score_multiplier_maximum_value": 3
                }
            }
        ],
        "numeric_fields_boosting": [
            {
                "input_value_config": {
                    "field_name": "centrality_score_15",
                    "default_field_value_if_missing": 0
                },
                "boost_function_config": {
                    "boost_function_input_value_rescaling_factor": 1,
                    "boost_function_increasing_function_type": "none"
                },
                "scoring_factor_config": {
                    "score_multiplier_constant_weight": 1,
                    "score_multiplier_variable_boost_weight": 2,
                    "score_multiplier_minimum_value": 1,
                    "score_multiplier_maximum_value": 10
                }
            }
        ],
        "categorical_matching": [
            {
                "categorical_field_name": "article_centroid_tags",
                "values_list": [
                    "centroid_k5", "centroid_k10", "centroid_k20", "centroid_k50", 
                    # "centroid_k20", 
                ],
                "score_multiplier": 1.5,
                "minimum_should_match": 1,
                "required": require_identified_as_centroid
            }
        ],
        "embedding_fields_to_return": [],  # don't return embeddings to reduce data transfer and latency 
        "limit": limit,
        "start": 0
    }

    if True:
        sentence_embedding = embedding_model.encode(q).tolist()
        request_json["embedding_fields_boosting"] = [
            {
                "input_value_config": {
                    "field_name": "title_embedding",
                    "target_embedding": sentence_embedding,
                    "default_embedding_comparison_value_if_missing": 0,
                    "embedding_comparison_metric": "cosine_similarity"
                },
                "boost_function_config": {
                    "boost_function_input_value_rescaling_factor": 1,
                    "boost_function_increasing_function_type": "none"
                },
                "scoring_factor_config": {
                    "score_multiplier_constant_weight": 3,
                    "score_multiplier_variable_boost_weight": 1,
                    "score_multiplier_minimum_value": 0,
                    "score_multiplier_maximum_value": 5
                }
            }
        ]

    if min_date != "" or max_date != "":
        date_filter = {
            "input_value_config": {
                "field_name": "publish_date",
            }
        }
        if min_date != "":
            date_filter["minimum_value"] = min_date + " 00:00:00"
        if max_date != "":
            date_filter["maximum_value"] = max_date + " 23:59:59"
        request_json["date_fields_filtering"] = [date_filter]

    search_response = open_search_rec.search(index_name, SearchConfig.parse_obj(request_json))
    hits = search_response.results
    print(f"len(hits) = {len(hits)} before fuzzy filtering")

    if (require_one_term_fuzzy_match or require_all_terms_fuzzy_match) and len(hits) > 0 and len(q) > 0:
        if perform_fuzzy_matching_on_title_only:
            hit_texts = [
                hit.item.get("text_fields", {}).get("article_title", "") 
                for hit in hits
            ]
        else:
            hit_texts = [
                # hit.item.get("text_fields", {}).get("article_title", "") #+ hit.item.get("text_fields", {}).get("article_text_beginning", "")
                hit.item.get("text_fields", {}).get("article_title", "") + hit.item.get("text_fields", {}).get("article_text_beginning", "")
                for hit in hits
            ]

        text_match_best_token_scores = []
        text_match_average_token_scores = []
        text_match_best_scores_per_query_token = []
        for text in hit_texts:
            text_tokens = [rapidfuzz.utils.default_process(text_token) for text_token in text.strip().split(" ")]
            query_tokens = [rapidfuzz.utils.default_process(query_token) for query_token in q.strip().split(" ")]
            best_scores_per_query_token = []
            for query_token in query_tokens:
                best_score = 0
                for text_token in text_tokens:
                    score = rapidfuzz.distance.JaroWinkler.similarity(query_token, text_token)
                    best_score = max(best_score, score)
                best_scores_per_query_token.append(best_score)
            text_match_best_scores_per_query_token.append(best_scores_per_query_token)
            text_match_best_token_scores.append(max(best_scores_per_query_token))
            text_match_average_token_scores.append(np.mean(best_scores_per_query_token))

        if require_all_terms_fuzzy_match:
            # print("text_match_best_scores_per_query_token", text_match_best_scores_per_query_token)
            valid_idxs = [idx for idx, best_scores_per_query_token in enumerate(text_match_best_scores_per_query_token) if np.min(best_scores_per_query_token) > 0.9]
        elif require_one_term_fuzzy_match:
            valid_idxs = [idx for idx, score in enumerate(text_match_best_token_scores) if score > 0.9]

        filtered_hits = [hit for idx, hit in enumerate(hits) if idx in valid_idxs]

        hits = filtered_hits

    return hits


@app.route('/search')
def search():
    q = request.args.get("q", "")[:250]  # max query length
    search_type = request.args.get("search_type", "timeline")
    ordering = request.args.get("ordering", "relevance_sort")
    min_date = request.args.get("min_date", "")
    max_date = request.args.get("max_date", "")
    target_date_boost = request.args.get("target_date_boost", "")


    matching_strictness = request.args.get("matching_strictness", "strict_matching")

    message = ""

    max_num_results = 500
    
    if len(q) > 0:
        try:
            if matching_strictness == "very_strict_matching":
                hits = \
                    perform_search(
                        open_search_rec, q, min_date, max_date, target_date_boost, 
                        require_all_terms_exact_match=True,
                        require_all_terms_fuzzy_match=True,
                        require_identified_as_centroid=False,
                        require_one_term_exact_match=False,
                        require_one_term_fuzzy_match=False,
                        perform_fuzzy_matching_on_title_only=True,
                        limit=max_num_results)
            elif matching_strictness == "strict_matching":
                if len(q.split()) > 1:
                    require_one_term_exact_match=True # if more than 1 search term, require at least 1 exact match
                else:
                    require_one_term_exact_match=False
                hits = \
                    perform_search(
                        open_search_rec, q, min_date, max_date, target_date_boost, 
                        require_all_terms_exact_match=False,
                        require_all_terms_fuzzy_match=True,
                        require_identified_as_centroid=False,
                        require_one_term_exact_match=require_one_term_exact_match,
                        require_one_term_fuzzy_match=False,
                        perform_fuzzy_matching_on_title_only=False,
                        limit=max_num_results)
            elif matching_strictness == "lenient_matching":
                hits = \
                    perform_search(
                        open_search_rec, q, min_date, max_date, target_date_boost, 
                        require_all_terms_exact_match=False,
                        require_all_terms_fuzzy_match=False,
                        require_identified_as_centroid=False,
                        require_one_term_exact_match=False,
                        require_one_term_fuzzy_match=True,
                        perform_fuzzy_matching_on_title_only=False,
                        limit=max_num_results)
            elif matching_strictness == "very_lenient_matching":
                hits = \
                    perform_search(
                        open_search_rec, q, min_date, max_date, target_date_boost, 
                        require_all_terms_exact_match=False,
                        require_all_terms_fuzzy_match=False,
                        require_identified_as_centroid=False,
                        require_one_term_exact_match=False,
                        require_one_term_fuzzy_match=False,
                        perform_fuzzy_matching_on_title_only=False,
                        limit=max_num_results)
            else:
                hits = []

            if len(hits) < 5 and matching_strictness != "very_lenient_matching":
                message = "You can reduce the matching strictness in order to get more results."
                
        except Exception as e:
            print(e)
            hits = []
            message = "No results due to an error."
    else:
        hits = []

    print(f"len(hits) = {len(hits)} with matching_strictness = {matching_strictness}")


    if search_type == "timeline":
        hits = [hit for hit in hits if hit.item["date_fields"]["publish_date"] is not None]


    # show older in top results too even if there are recent news with higher score (due to recency boosting)
    recent_threshold = datetime.utcnow() - timedelta(days=7)
    recent_results = [hit for hit in hits if hit.item["date_fields"]["publish_date"] >= recent_threshold ]
    recent_results_in_top_20 = [hit for hit in hits[:20] if hit.item["date_fields"]["publish_date"] >= recent_threshold ]
    older_results = [hit for hit in hits if hit.item["date_fields"]["publish_date"] < recent_threshold ]
    num_recent_results = len(recent_results)
    num_older_results = len(older_results)
    num_recent_results_in_top_20 = len(recent_results_in_top_20)
    if num_recent_results_in_top_20 > 5:
        hits = merge_lists([recent_results, older_results], list_frequencies=[1,3])
    
    for relevance_rank, hit in enumerate(hits):
        hit.item["extra_information"]["relevance_rank"] = relevance_rank

    print(f"len(hits) = {len(hits)}")
    if search_type == "timeline":
        hits = sorted(hits, reverse=True, key=lambda hit: hit.item["date_fields"]["publish_date"])
    
    print(f"len(hits) = {len(hits)}")

    if search_type == "timeline":
        initial_number_of_results_to_show=min(100, len(hits))
    else:
        initial_number_of_results_to_show = len(hits)

    search_type_timeline = "checked" if search_type == "timeline" else ""
    search_type_news_search = "checked" if search_type == "news_search" else ""


    very_strict_matching = "checked" if matching_strictness == "very_strict_matching" else ""
    strict_matching = "checked" if matching_strictness == "strict_matching" else ""
    lenient_matching = "checked" if matching_strictness == "lenient_matching" else ""
    very_lenient_matching = "checked" if matching_strictness == "very_lenient_matching" else ""


    return render_template(
        "search.html", q=q, hits=hits,
        min_date=min_date, max_date=max_date,
        target_date_boost=target_date_boost,
        search_type_timeline=search_type_timeline,
        search_type_news_search=search_type_news_search,
        num_results=len(hits),
        initial_number_of_results_to_show=initial_number_of_results_to_show,
        very_strict_matching=very_strict_matching,
        strict_matching=strict_matching,
        lenient_matching=lenient_matching,
        very_lenient_matching=very_lenient_matching,
        message=message,
        debugging=debugging
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
