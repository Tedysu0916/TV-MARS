from asdTools.Classes.Base.BaseModel import BaseModel
from asdTools.Classes.Tool.GitIgnore import GitIgnore


class SaveExpData(BaseModel):
    # before use: pip install gitignore-parser
    def __init__(self, exp_dir:str="", max_size:float=0, unit:str="MB", gitignore:str="", ext_exclude:list=[], **kwargs) -> None:
        super().__init__(multipleFiles=True, **kwargs)
        self.exp_dir = exp_dir
        self.max_size = max_size
        self.unit = unit
        self.ext_exclude = ext_exclude
        self.gitignore = GitIgnore(gitignore)

    def __call__(self, exp_dir:str="", max_size:float=0, unit:str="", ext_exclude:list=[]):
        return self.run(exp_dir, max_size, unit, ext_exclude)

    def run(self, exp_dir:str="", max_size:float=0, unit:str="", ext_exclude:list=[]):
        if exp_dir == "":
            exp_dir = self.exp_dir
        if max_size == 0:
            max_size = self.max_size
        if unit == "":
            unit = self.unit
        if ext_exclude == []:
            ext_exclude = self.ext_exclude
        # all files
        files_path = self.get_paths_from_dir(exp_dir)
        self.log(f"{len(files_path)} files found")
        files_path = self.gitignore(files_path)
        self.log(f"{len(files_path)} files after .gitignore (if have)")
        files_path = self.filter_ext(files_path, ext_exclude, False)
        self.log(f"{len(files_path)} files after exclude files with extension in {ext_exclude}")
        files_small = self.filter_size(files_path, max_size, unit)
        files_large = set(files_path).difference(set(files_small))
        self.separator("Small File")
        for i, file in enumerate(files_small):
            file_with_dir = self.remove_root_of_path(file, exp_dir)
            output_path = self.generate_output_path(output_file=file_with_dir)
            self.copy(file, output_path)
            self.log(f"{i+1}: {file} --copy-> {output_path}")
        self.separator("Large File")
        for i, file in enumerate(files_large):
            self.log(f"{i+1}: {round(self.get_size_of_file(file), 2)}{unit}\t {file} too large")
        message = []
        message.append("感谢mherrmann实现的gitignore_parser，但它的效果有点不好，这里做了一点改进，但是还有部分问题无法解决")
        message.append("请注意核对结果！")
        message.append("Thanks to mherrmann for implementing gitignore_parser, but its effectiveness is not ideal. Some improvements have been made here, but there are still some issues that cannot be resolved.")
        message.append("ATTENTION: review the results!")
        message.append("SCI +1  d(`･∀･)b")
        self.done(message)


if __name__ == "__main__":
    # 因为仅复制小文件，因此可以先将max_size设小，根据日志中的大文件信息来逐步调整，以获得更好的保存效果。
    # Since only small files are being copied, you can start by setting the max_size to a small value and gradually adjust it based on the information about large files in the logs to achieve better storage efficiency.
    # before use: pip install gitignore-parser
    
    """
    max_size:       小于max_size的文件将被保留
                    the file small than max_size will be preserved.
    unit:           max_size单位，默认为MB，支持b, B, KB, MB, GB
                    the unit of max_size, default in "MB", support with b, B, KB, MB, GB
    ext_exclude[]:  列表，文件后缀存在于列表时，将被排除
                    list, file will be exclude if its extension in ext_exclude[]
    gitignore:      .gitignore
    """
    
    # SaveCVExp("./", max_size=1, unit="MB", gitignore=".gitignore")()
    exp_dir = r"Sample/ColorGT"
    SaveExpData(exp_dir, max_size=1, unit="KB", ext_exclude=["jpg", "bmp", 'png'])()





        