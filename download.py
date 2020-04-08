from utils.download_subreddit import DownloadSubreddit


if __name__ == "__main__":
    reddit_downloader = DownloadSubreddit(
        subreddit="nosleep",
        option="submission",
        start=1554768000,
        least_num_comments=3,
        path="data/",
    )
    reddit_downloader.download()
