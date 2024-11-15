from asdTools.Classes.Base.BaseModel import BaseModel
from bs4 import BeautifulSoup
import html2text
import requests
import urllib
import re



class WordpressSpiderInMarkdown(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.watermark = "> 文章首发见博客：[{post_url}]({post_url})。\n"
        self.watermark += "> 无图/格式错误/后续更新请见首发页。\n"
        self.watermark += "> 更多更新请到[mwhls.top](https://mwhls.top)查看\n"
        self.watermark += "> 欢迎留言提问或批评建议，私信不回。\n\n"

    def crawl(self, url):
        head = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"}
        request = urllib.request.Request(url, headers=head)
        try:
            response = urllib.request.urlopen(request)
            html = response.read().decode("utf-8")
        except urllib.error.URLError as e:
            if hasattr(e, "code"):
                self.raise_error(e.code)
            if hasattr(e, "reason"):
                self.raise_error(e.reason)
        return html

    def get_post(self, url):
        # crawl html
        html = self.crawl(url)

        # ---------- get title ----------
        # init
        soup = BeautifulSoup(html, "html.parser")
        title = soup.find('h1', class_="post-title")
        title = title.text[1:]
        
        # ---------- parse content ----------
        # init
        soup = BeautifulSoup(html, "html.parser")
        # raw data
        post_content = soup.find('div', class_="post-content-content")
        # remove catalog
        try:
            catalog_to_remove = soup.find('div', class_='lwptoc lwptoc-baseItems lwptoc-inherit')
            if catalog_to_remove:   
                catalog_to_remove.extract()
        except:
            pass
        # remove rount of reading
        count_to_remove = soup.find('div', class_='post-views')
        if count_to_remove:
            count_to_remove.extract()
        # remove rount of post feature image
        count_to_remove = soup.find('picture')
        if count_to_remove:
            count_to_remove.extract()
        # code mark
        code_blocks = soup.find_all('pre')
        codes = []
        try:
            for code_block in code_blocks:
                code_html = str(code_block)
                pattern = r'<pre><code class="language-(.*?)">'
                code_type = re.search(pattern, code_html).group(1)
                code_markdown = f"```{code_type}"
                code_markdown += html2text.html2text(code_html)[:-1]
                code_markdown += "```"
                codes.append(code_markdown)
                code_block.replace_with("MWHLS_CODE-BLOCK-MARK_MWHLS")
        except:
            pass
        # html2markdown
        post_markdown = html2text.html2text(str(post_content))
        # replace code
        for code in codes:
            post_markdown = post_markdown.replace("MWHLS_CODE-BLOCK-MARK_MWHLS", code, 1)
        
        # ---------- water mark ----------
        # add water mark
        result = self.watermark.replace("{post_url}", url)
        result += post_markdown
        return title, result
