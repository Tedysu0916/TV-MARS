from gitignore_parser import parse_gitignore
import tempfile
import time
import os


class GitIgnore():
    # before use: pip install gitignore-parser
    # 感谢mherrmann实现的gitignore_parser，但它的效果有点不好，这里做了一点改进，但是还有部分问题无法解决
    # 请注意核对结果！
    # Thanks to mherrmann for implementing gitignore_parser, but its effectiveness is not ideal. Some improvements have been made here, but there are still some issues that cannot be resolved.
    # ATTENTION: review the results!
    def __init__(self, path:str=""):
        self.tmpfile = "temp_gitignore_asdTools-" + str(time.time())
        self.gitignore_path = path

    def __call__(self, files:list):
        if isinstance(files, list):
            return self.fliter_gitignore(files)
        else:
            return self.check_ignore(files)
        
    def generate_gitignore(self, path:str):
        with open(self.tmpfile, mode='w', encoding="utf8") as temp_file:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    temp_file.write(line + "\n")
                    if line[-1] == "/":
                        for i in range(10):
                            line = line + "*/"
                            temp_file.write(line + "\n")
                temp_file.write("temp_gitignore_asdTools-*" + "\n")
        gitignore = parse_gitignore(self.tmpfile)
        self.gitignore = gitignore


    def check_ignore(self, file:str) -> bool:
        try:
            isIgnore = self.gitignore(file) 
        except:
            isIgnore = False
        return isIgnore
    
    def fliter_gitignore(self, files:list) -> list:
        if self.gitignore_path == "":
            return files
        else:
            self.generate_gitignore(self.gitignore_path)
        res = []
        # abandon = []
        for file in files:
            if not self.check_ignore(file):
                res.append(file)
            # else:
            #     abandon.append(file)
        os.remove(self.tmpfile)
        return res