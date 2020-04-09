from utils.download_subreddit import DownloadSubreddit

if __name__ == "__main__":
    reddit_downloader = DownloadSubreddit(
        subreddit="nosleep",
        option="submission",
        start=1523318400,
        least_num_comments=5,
        path="dataset/",
        use=""
    )
    reddit_downloader.download()
    reddit_downloader.split()
