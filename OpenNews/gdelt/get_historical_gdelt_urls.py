"""
    Generate a CSV file containing a list of URLs from the source_domains listed in the config files using GDELT
"""


import argparse
import datetime
import gzip
import logging
import os
import pandas as pd
import requests
import traceback
import json


def getDomain(url_):
    url = url_.lower()
    url = url.strip()
    url = url.replace("https://", "")
    url = url.replace("http://", "")
    url = url.replace("www.", "")
    if url.find("/") != -1:
        url = url[:url.find("/")]
    if url.find("?") != -1:
        url = url[:url.find("?")]
    return url


def normalizeUrl(url):
    url = url.replace("https://", "")
    url = url.replace("http://", "")
    if url.find("?") != -1:
        url = url[:url.find("?")]
    if url[-1] == "/":
        url = url[:-1]
    return url


def getFileNameForUrl(year, month, day, hour, url_suffix="0000.LINKS.TXT.gz"):
    month_str = str(month)
    if month < 10:
        month_str = "0" + month_str

    day_str = str(day)
    if day < 10:
        day_str = "0" + day_str

    hour_str = str(hour)
    if hour < 10:
        hour_str = "0" + hour_str

    return str(year) + month_str + day_str + hour_str + url_suffix


def getLinkFileUrl(year, month, day, hour, 
                   base_url="http://data.gdeltproject.org/gdeltv3/gfg/alpha/",
                   url_suffix="0000.LINKS.TXT.gz"):

    month_str = str(month)
    if month < 10:
        month_str = "0" + month_str

    day_str = str(day)
    if day < 10:
        day_str = "0" + day_str

    hour_str = str(hour)
    if hour < 10:
        hour_str = "0" + hour_str

    #return base_url + str(year) + month_str + day_str + hour_str + url_suffix
    return base_url + getFileNameForUrl(year, month, day, hour)


def getDomainNames(file_path):
    with open(file_path) as f:
        domains = f.read().split("\n")
    domains = list(map(lambda domain: domain if domain.find(" ") == -1 else domain[:domain.find(" ")], domains))
    domains = list(filter(lambda domain: domain is not None and len(domain) > 0, domains))
    return domains



def downloadGdeltFile(gdelt_file_url, logger, target_file_path="tmp_gdelt_file.txt.gz"):
    if not os.path.isfile(target_file_path):
        logger.info(f"Downloading  {gdelt_file_url}")
        gdelt_response = requests.get(gdelt_file_url, allow_redirects=False, timeout=60*5)
        gdelt_content = gdelt_response.content
        with open(target_file_path, 'wb') as f:
            f.write(gdelt_response.content)
    else:
        logger.info(f"{target_file_path} file already exists")
    return 


def getUrlsFromGdeltFileUrl(gdelt_file_url, source_domain_list, logger, target_file_path="tmp_gdelt_file.txt.gz"):
    logger.info("getting file lines")
    with gzip.open(target_file_path, "rb") as f:
        lines = f.readlines()
    
    logger.info("processing file lines")
    all_urls = set()
    for line in lines[:]:
        url = line.split(b"\t")[4].decode("utf-8")
        all_urls.add(url)
    
    logger.info(f"number of urls in file = {len(all_urls)}")

    filtered_urls = []
    for url in all_urls:
        if getDomain(url) in source_domain_list:
            #print("url =", url)
            filtered_urls.append(url)

    logger.info(f"number of urls in file with domain name in list = {len(all_urls)}")

    normalized_filtered_urls = []
    for url in filtered_urls:
        try:
            normalized_filtered_urls.append(normalizeUrl(url))
        except Exception as e:
            print("\n")
            print(e)
            traceback.print_exc()
            print("url =", url)
            print("\n")

    return normalized_filtered_urls


if __name__ == "__main__":
    print("get_articles_url_list.py")

    loggin_level = logging.INFO

    logger = logging.getLogger('logger')
    logging_handler = logging.StreamHandler()
    logging_handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
    logger.addHandler(logging_handler)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--download_file_base_path', type=str, default="OpenNews/gdelt/data/frontpage/")
    parser.add_argument('--url_file_path', type=str, default="OpenNews/gdelt/data/urls.csv")
    parser.add_argument('--domain_list_filepath', type=str, default="OpenNews/gdelt/gdelt_domains.txt")
    parser.add_argument('--keep_downloaded_gdelt_files', default=False, action="store_true")
    parser.add_argument('--start_year', type=int, default=2022)
    parser.add_argument('--num_skip_hours', type=int, default=12, help="6, 12 or 24")

    args = parser.parse_args()
    logger.info(args)

    num_skip_days = 1 # you can change this but leaving to 1 makes sense


    min_hour = 0
    max_hour = 23
    min_day = 1
    max_day = 31
    min_month = 1
    max_month = 12
    
    now = datetime.datetime.utcnow()

    end_year = now.year


    with open(args.domain_list_filepath, "r") as f:
        domain_list = f.readlines()

    domain_list = [getDomain(d) for d in domain_list]
    print("domain_list = {}".format(domain_list), "\n")
    print("len(domain_list) = {}".format(len(domain_list)), "\n")


    ###############

    urls = set()
    for year in range(args.start_year, end_year+1):
        for month in range(min_month, max_month+1):
            for day in range(min_day, max_day+1, num_skip_days):
                for hour in range(min_hour, max_hour+1, args.num_skip_hours):
                    logger.info("year = {}, month = {}, day = {}, hour = {}".format(year, month, day, hour))

                    target_file_path = args.download_file_base_path + getFileNameForUrl(year, month, day, hour, url_suffix="0000.LINKS.TXT.gz")

                    gdelt_url = getLinkFileUrl(year, month, day, hour)
                    logger.info("gdelt_url ="+ gdelt_url)

                    if not os.path.isfile(target_file_path):
                        logger.info(f"downloading gdelt file")
                        downloadGdeltFile(gdelt_url, logger, target_file_path=target_file_path)
                    else:
                        logger.info(f"gdelt file already downloaded")


                    num_prior_urls = len(urls)
                    logger.info(f"processing file {gdelt_url}")
                    domain_urls = getUrlsFromGdeltFileUrl(gdelt_url, domain_list, logger, target_file_path=target_file_path)
                    logger.info(f"len(domain_urls) = {len(domain_urls)}")
                    urls = urls.union(set(domain_urls))
                    logger.info(f"num new urls found = {len(urls) - num_prior_urls}")


                    if not args.keep_downloaded_gdelt_files:
                        logger.info("removing file")
                        os.remove(target_file_path)

                    logger.info(f"len(urls) = {len(urls)} \n")

    logger.info(f"len(urls) = {len(urls)}")


    domain_names = list(map(lambda url: getDomain(url), urls))

    url_df = pd.DataFrame(data={"url":list(urls), "domain_name": domain_names})

    url_df.to_csv(args.url_file_path)