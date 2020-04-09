from sklearn.model_selection import train_test_split
from datetime import datetime
import requests
import json


class DownloadSubreddit:
    def __init__(
        self,
        subreddit,
        option="submission",
        start=0,
        least_num_comments=3,
        path="data/",
        use="train"
    ):
        self.subreddit = subreddit
        self.option = option
        self.start = start
        self.data = []
        self.path = path
        self.use = use
        self.least_num_comments = least_num_comments
        self.url = "http://api.pushshift.io/reddit"

    def fetch(self, **kwargs):
        params = {"sort_type": "created_utc", "sort": "asc", "size": 1000}
        params["subreddit"] = self.subreddit
        params["option"] = self.option
        params["after"] = kwargs["after"]
        print(params)
        r = requests.get(
            self.url + "/" + self.option + "/search/", params=params, timeout=30
        )
        if r.status_code == 200:
            response = json.loads(r.text)
            data = response["data"]
            sorted_data_by_id = sorted(data, key=lambda x: int(x["id"], 36))
            return sorted_data_by_id

    def download(self, **kwargs):
        max_created_utc = self.start
        max_id = 0
        filename = self.path + self.subreddit + "_" + self.use
        file = open(f"{filename}.txt", "w+")
        count = 0
        start_date = datetime.fromtimestamp(self.start).strftime("%Y-%m-%d %H:%M:%S")
        end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
                        text = object["selftext"]
                        num_comments = object["num_comments"]
                        if (
                            text == "[removed]"
                            or text == "[deleted]"
                            or text == ""
                            or num_comments < self.least_num_comments
                        ):
                            continue
                        else:
                            count += 1
                            file.write(f"<|startoftext|> " + text + " <|endoftext|>\n")
                            # self.data.append(text + " <|endoftext|>")
                    except:
                        continue
            if nothing_processed:
                file.close()
                print(
                    f"{count} stories from {start_date} to {end_date} with at least {self.least_num_comments} comments were fetched."
                )
                return
            max_created_utc -= 1

    def split(self):
        story_list = []
        with open(self.path + self.subreddit + "_" + self.use + ".txt", "r") as s:
            story_reader = s.read().split("<|startoftext|> ")
            for story in story_reader:
                story_list.append(story)
        num_train = int(0.8 * len(story_list))
        train = story_list[:num_train]
        evaluate = story_list[num_train:]
        with open(f"data/{self.subreddit}_train.txt", "w+") as train_file:
            for story in train:
                train_file.write(story)
            train_file.close()
        with open(f"data/{self.subreddit}_evaluate.txt", "w+") as evaluate_file:
            for story in evaluate:
                evaluate_file.write(story)
            evaluate_file.close()

