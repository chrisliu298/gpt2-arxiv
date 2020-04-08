from utils.download_reddit_data import DataPipeline

reddit_downloader = DataPipeline("nosleep", "submission", start=1577836800)
reddit_downloader.download()
