from asdTools.Classes.Tool.FileManager import FileManager
from asdTools.Classes.Tool.Counter import CounterLoop
from asdTools.Classes.File.DocBase import DocBase


class DocReader(DocBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.file_reader = FileManager(self._log_dir)

    def add_file(self, path:str):
        self.file_reader.add(path)

    def add_files(self, paths:list):
        for path in paths:
            self.add_file(path)

    def fliter_doc(self, paths):
        docs = []
        for i, path in enumerate(paths):
            if self.check_ext(path, ["docx", "doc", "pdf"]):
                docs.append(path)
        return docs
    
    def get_current_doc(self) -> dict:
        doc = self.file_reader.get_current()
        return doc

    def get_previous_doc(self, step:int=1):
        doc = self.file_reader.get_previous(step)
        return doc

    def get_next_doc(self, step:int=1):
        doc = self.file_reader.get_next(step)
        return doc

    def read_as_pdf(self, doc:dict) -> list:
        if doc["type"] in ["doc", "docx", "pdf"]:
            read_fn = self.read_docxs
        content = self.file_reader.read(doc, read_fn)
        return content