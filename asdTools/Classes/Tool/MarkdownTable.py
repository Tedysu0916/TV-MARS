import os


class MarkdownTable():
    def __init__(self, title:list):
        self.table = ""
        self.make_title(title)

    def __call__(self, row:list):
        self.add_row(row)

    def add_row(self, row:list):
        self.table += "|"
        for i in row:
            self.table += " " + str(i) + " |"
        self.table += "\n"

    def convert_imgPath_to_MDImgPath(self, img_path:str, name:str=""):
        if name == "":
            name = os.path.basename(img_path)
        return f"![{name}]({img_path})"

    def make_title(self, title:list):
        self.table += "|"
        for i in title:
            self.table += " " + i + " |"
        self.table += "\n"
        self.table += "|"
        for i in title:
            self.table += " --- |"
        self.table += "\n"

    def output(self):
        return self.table
    
if __name__ == "__main__":
    md_table = MarkdownTable(["title1", "title2", "title3"])
    img = r"F:\0_DATA\1_DATA\Study\202308_CCReID\Image\C_cropped_rgb094.jpg"
    img = md_table.convert_imgPath_to_MDImgPath(img)
    md_table(["a", img, "c"])
    from asdTools.Classes.Base.BaseModel import BaseModel
    logger = BaseModel()
    save_path = logger.generate_output_path(output_file="test.md")
    logger.save_file(md_table.output(), save_path)
