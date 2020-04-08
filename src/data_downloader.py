from utils.download_reddit_data import DataPipeline

reddit_downloader = DataPipeline(
    "nosleep", "submission", start=1554768000, least_num_comments=3
)
reddit_downloader.download()
