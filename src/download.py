import requests
import json
import re
import time

PUSHSHIFT_REDDIT_URL = "http://api.pushshift.io/reddit"


def fetch_objects(**kwargs):
    params = {"sort_type": "created_utc", "sort": "asc", "size": 1000}
    for key, value in kwargs.items():
        params[key] = value
    print(params)
    type = "comment"
    if "type" in kwargs and kwargs["type"].lower() == "submission":
        type = "submission"

    r = requests.get(
        PUSHSHIFT_REDDIT_URL + "/" + type + "/search/", params=params, timeout=30
    )

    if r.status_code == 200:
        response = json.loads(r.text)
        data = response["data"]
        sorted_data_by_id = sorted(data, key=lambda x: int(x["id"], 36))
        return sorted_data_by_id


def extract_reddit_data(**kwargs):
    max_created_utc = 1577836800
    max_id = 0
    filename = kwargs["subreddit"]
    file = open(f"{filename}.txt", "w+")
    while 1:
        nothing_processed = True
        objects = fetch_objects(**kwargs, after=max_created_utc)
        for object in objects:
            id = int(object["id"], 36)
            if id > max_id:
                nothing_processed = False
                created_utc = object["created_utc"]
                max_id = id
                if created_utc > max_created_utc:
                    max_created_utc = created_utc
                try:
                    text = object["selftext"]
                    if text == "[removed]" or text == "":
                        continue
                    else:
                        file.write("<|startoftext|> ")
                        file.write(text.replace("\n", "") + " <|endoftext|>\n")
                except:
                    continue
        if nothing_processed:
            return
        max_created_utc -= 1
    file.close()


if __name__ == "__main__":
    extract_reddit_data(subreddit="nosleep", type="submission")
