# OpenNewsSearch

This is an early prototype of OpenNewsSearch, an open source news search and aggregation engine licensed under the GPL-3.0 license.
There are 2 reasons for choosing this license:
First, to encorage people to contribute to this open source project. 
Second, for transparency, modifications have to be made public.


# Dev deployment using docker-compose

## Set environment variables
```
export opensearchrec_database_type=elasticsearch
export opensearchrec_database_url=http://localhost:9200
export opensearchrec_database_username=elastic
export opensearchrec_database_password=CHANGE_THIS
export elasticsearch_verify_certificates=False
export debugging=False
export code_url=https://github.com/OpenSearchRec/OpenNewsSearch
```

## Start Dev ElasticSeach
```
mkdir elasticsearch
docker compose -f docker-compose-dev-elasticsearch.yml up -d
```

## Start News Crawler and Analyzer
```
docker compose -f docker-compose-recent-news-crawling.yml build
docker compose -f docker-compose-recent-news-crawling.yml up -d
```
and then
```
docker compose -f docker-compose-recent-news-analysis.yml build
docker compose -f docker-compose-recent-news-analysis.yml up -d

```

## Start Flask App
```
docker compose -f docker-compose-flask-app.yml build
docker compose -f docker-compose-flask-app.yml up -d
```

# Load some historical news data

## Install Dependencies
```
python3 -m venv env
. env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Getting Historical News Article URLs
```
mkdir OpenNews/gdelt/data
mkdir OpenNews/gdelt/data/frontpage
python -m OpenNews.gdelt.get_historical_gdelt_urls --start_year=2020 --url_file_path=OpenNews/gdelt/data/urls_2020_.csv --num_skip_hours=12 --keep_downloaded_gdelt_files
```


## Crawling and Indexing List of Article URLs
```
python -m OpenNews.ingestion.crawl_and_index_articles_from_url_list --article_url_filepath=OpenNews/gdelt/data/urls_2020_.csv  --allowed_language_codes=en --opensearchrec_news_index_name=news_us_en --num_processes=2

```

## Running the analysis on the historical news
```
python -m OpenNews.analysis.news_analysis --max_num_days_back=1095 --repeat_analysis_in_infinite_loop --config_filepath=configs/open_news_config.yml
```
