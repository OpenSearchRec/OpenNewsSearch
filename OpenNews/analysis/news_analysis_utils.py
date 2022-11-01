import argparse
from datetime import datetime, timedelta
import logging
import os
import yaml

import numpy as np
from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


from OpenSearchRec.post_ranking import (
    select_items_greedily_and_with_similarity_suppression,
    select_top_k_centroids
)



use_generic_news_data_for_idf = True

if use_generic_news_data_for_idf:
    from sklearn.datasets import fetch_20newsgroups
    news_data, _ = fetch_20newsgroups(return_X_y=True)
    generic_news_data = list(news_data)



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

