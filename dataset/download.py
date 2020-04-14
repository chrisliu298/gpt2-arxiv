from utils.download_subreddit import DownloadSubreddit

if __name__ == "__main__":
    reddit_downloader = DownloadSubreddit(
        subreddit="shortscarystories",
        option="submission",
        start=1491696000,
        least_num_comments=3,
        path="dataset/",
        use=""
    )
    reddit_downloader.download()
    reddit_downloader.split()
