from asdTools.Classes.Base.BaseModel import BaseModel
import sys


class SDStylesBeautify(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, path:str) -> str:
        return self.run(path)

    def add_category_head(self, content:list, delimter='_'):
        category_heads = []
        for row in range(len(content)):
            i = row + len(category_heads)
            col1 = content[i][0]
            col1_split = col1.split(delimter)
            if len(col1_split) >= 2:
                category_head = col1_split[0]
                if category_head not in category_heads:
                    category_heads.append(category_head)
                    category_row = [''] * len(content[i])
                    category_row[0] = f"---{category_head}---"
                    content.insert(i, category_row)
        return content

    def run(self, path:str) -> str:
        content = self.read_csvLike(path)
        content = self.filter_csv_empty(content)
        content = self.remove_category_head(content)
        content = self.sort_csv_by_col1(content)
        content = self.add_category_head(content)
        backup_file = f"styles_{self.get_time(True)}.csv"
        backup_path = self.generate_output_path(output_file=backup_file)
        self.copy(path, backup_path)
        self.log(f"Original styles.csv have backup from {path} to {backup_path}", logWhenDone=True)
        output_dir = self.get_dir_of_file(path)
        output_file = self.get_name_of_file(path, True)
        output_path = self.save_csv(content, output_dir=output_dir, output_file=output_file)
        self.log(f"Styles.csv has been beatufied: {output_path}.", logWhenDone=True)
        self.done()
        return output_path
        
    def remove_category_head(self, content:list) -> list:
        category_heads = []
        for row in range(len(content)):
            i = row - len(category_heads)
            col1 = content[i][0]
            if col1[:3] == '---' and col1[-3:] == '---':
                category_heads.append(col1[3:-3])
                del content[i]
        return content


# if __name__ == "__main__":
#     path = r"F:\0_DATA\1_DATA\CODE\PYTHON\0_StableDiffusion\sd-webui-aki\sd-webui-aki-v4/styles.csv"
#     SD_styles_beautify = SDStylesBeautify()
#     SD_styles_beautify(path)

    
if __name__ == "__main__":
    """
    
    """
    try:
        csv_path = sys.argv[1]
        csv_path = csv_path.strip('"')
        SD_styles_beautify = SDStylesBeautify()

        output_path = SD_styles_beautify(csv_path)
        SD_styles_beautify.pause()
    except Exception as e:
        print(e)
        input("Press Enter to exit.")