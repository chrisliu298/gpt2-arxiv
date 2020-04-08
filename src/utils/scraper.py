import requests
import json
import re
import time


class Scraper:
    def __init__(self, subreddit, option="submission"):
        self.subreddit = subreddit
        self.option = option
        self.data = None
        self.url = "http://api.pushshift.io/reddit"

    def fetch(self, **kwargs):
        params = {"sort_type": "created_utc", "sort": "asc", "size": 1000}
        params['subreddit'] = self.subreddit
        params['option'] = self.option
        params['after'] = kwargs['after']
        print(params)
        r = requests.get(self.url + "/" + self.option + "/search/", params=params, timeout=30)
        if r.status_code == 200:
            response = json.loads(r.text)
            data = response["data"]
            sorted_data_by_id = sorted(data, key=lambda x: int(x["id"], 36))
            return sorted_data_by_id

    def extract(self, **kwargs):
        max_created_utc = 1577836800
        max_id = 0
        filename = self.subreddit
        file = open(f"{filename}.txt", "w+")
        while 1:
            nothing_processed = True
            objects = self.fetch(**kwargs, after=max_created_utc)
            for object in objects:
                id = int(object["id"], 36)
                if id > max_id:
                    nothing_processed = False
                    created_utc = object["created_utc"]
                    max_id = id
                    if created_utc > max_created_utc:
                        max_created_utc = created_utc
                    try:
                        text = object["selftext"].replace("\n", "")
                        if text == "[removed]" or text == "":
                            continue
                        else:
                            file.write("<|startoftext|> ")
                            file.write(text + " <|endoftext|>\n")
                    except:
                        continue
            if nothing_processed:
                return
            max_created_utc -= 1
        file.close()
