from asdTools.Classes.Spider.WordPressSpiderInMarkdown import WordpressSpiderInMarkdown
from asdTools.Classes.Tool.Clipboard import Clipboard

    
if __name__ == "__main__":
    """
    Before: https://mwhls.top/4810.html
    After: https://blog.csdn.net/asd123pwj/article/details/132394313
    仅在我的博客测试正常：mwhls.top
    爬取WordPress文章，并转为markdown格式
    Only testing in my blog: mwhls.top
    Crawl post of WordPress, and output in markdown
    """
    spider = WordpressSpiderInMarkdown()
    spider.log("Input post url from mwhls.top:") 
    url = spider.input("")

    spider.log(f"Parsing {url}")
    title, result = spider.get_post(url)

    spider.log("Title has copied")
    Clipboard.copy(title)
    spider.pause()

    spider.log("Content has copied")
    Clipboard.copy(result)
    spider.done()
    spider.pause()