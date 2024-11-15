from asdTools.Classes.Base.BaseModel import BaseModel
import html2text


class HTMLBase(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def convert_html_to_markdown(self, html):
        if self.exists(html):
            html = self.read_html(html)
        markdown = html2text.html2text(html)
        return markdown

if __name__ == "__main__":
    # no test
    html = r""
    htmlbase = HTMLBase()
    htmlbase.convert_html_to_markdown()