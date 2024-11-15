from asdTools.Classes.Base.BaseModel import BaseModel
from asdTools.Classes.Tool.Counter import CounterLoop

class FileManager(BaseModel):
    def __init__(self, 
                 empty_item:dict={
                        "id": "",
                        "path": "",
                        "name": "Empty item",
                        "type": {"file": "", "content": "", "analysis": ""},
                        "content": "",
                        "analysis": "",
                        "added": ""
                    },
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.config_file = "file_manage.config"
        config_path = self.generate_path(output_dir=self._log_dir, output_file=self.config_file)
        if self.exists(config_path):
            self.config = self.read_json(config_path)
            self.name2item = {file["name"]: file for _, file in self.config["files"].items()}
            self.recovery_config()
        else:
            self.config = {
                "created": self.get_time(),
                "modified": self.get_time(),
                "files": {},
                "order": []
            }
            self.name2item = {}
        self.counter = CounterLoop(self.config["files"], self.config["order"], 0)
        self.empty_item = empty_item

    def __len__(self):
        return len(self.config["files"])

    def add(self, path:str, content:str=""):
        if isinstance(path, str):
            # add from file or text
            file_md5 = self.get_md5_of_txt(content) if content else self.get_md5_of_file(path)
            if file_md5 in self.config["files"].keys():
                return self.config["files"][file_md5]
            item = self.deepcopy(self.empty_item)
            item["id"] = file_md5
            item["type"]["file"] = "txt" if content else self.get_ext_of_file(path)
            item["name"] = self.get_name_of_file(path) + f"_{self.get_time(True)}" + f".{item['type']['file']}"
            item["added"] = self.get_time()
            item["path"] = self.join(self._log_dir, item["type"]["file"], item["name"])
            if content:
                self.save_file(content, item["path"])
            else:
                self.copy(path, item["path"])
            self.name2item[item["name"]] = item
            self.config["files"][file_md5] = item
            self.config["modified"] = self.get_time()
            self.config["order"].append(file_md5)
            self.counter.update(self.config["files"], self.config["order"])
            # print(self.config["files"])
            # print(list(self.config["files"].keys()))
            # for _, item in self.config["files"].items():
            #     print(item["name"])
            self.save_config()
            return item
        elif isinstance(path, list):
            # add from files
            new = []
            for p in path:
                new.append(self.add(p))
            return new

    def analyse_item(self, content, type:str, file:dict):
        if type == "json":
            return "Analyse json", "json"
        else:
            return "No Analyse", "str"

    def analyse(self, file:dict, analyse_fn=None, save_fn=None, load_fn=None):
        if analyse_fn == None:
            analyse_fn = self.analyse_item
        if save_fn == None:
            save_fn = self.save_item
        if load_fn == None:
            load_fn = self.load_item
        analysis = file["analysis"]
        analysis_path = self.generate_storage_path(file, 1)
        if analysis == "":
            analysis, type = analyse_fn(file["content"], file["type"]["content"], file)
            # analysis_file = file["name"] + ".analysis"
            # analysis_path = self.join(self.log_dir, file["type"], analysis_file)
            file["analysis"] = analysis
            file["type"]["analysis"] = type
            save_fn(analysis, analysis_path, file["type"]["analysis"])
            self.save_config()
        else:
            load_fn(analysis_path, file["type"]["analysis"])
        # else:
        #     if load_fn:
        #         analysis = load_fn(analysis_path)
        #     else:
        #         analysis = self.read_json(analysis_path)
        return analysis

    def fileExists(self, path:str="", content:str=""):
        if content:
            file_md5 = self.get_md5_of_txt(content)
        elif path:
            file_md5 = self.get_md5_of_file(path)
        return True if file_md5 in self.config["files"].keys() else False

    def map_name_to_item(self, name:str):
        return self.name2item[name]

    def generate_storage_path(self, item:dict, mode:int, objects:list=["content", "analysis"]) -> str:
        obj = objects[mode]
        storage_dir = self.join(self._log_dir, item["type"]["file"])
        storage_file = item["name"] + "." + obj
        storage_path = self.generate_path(output_dir=storage_dir, output_file=storage_file)
        return storage_path

    def get_current(self) -> dict:
        res = self.counter.current()
        return res

    def get_previous(self, step:int=1):
        res = self.counter.previous(step)
        return res

    def get_next(self, step:int=1):
        res = self.counter.next(step)
        return res
    
    def load_item(self, path:str, type:str):
        if type in ["list", "str", "tuple"]:
            with open(path, "r", encoding="utf8") as f:
                content = f.readlines()
                content = "\n".join(content)
                content = content.strip()
        elif type in ["json", "dict"]:
            content = self.read_json(path)
        else:
            with open(path, "r", encoding="utf8") as f:
                content = f.readlines()
                content = "\n".join(content)
                content = content.strip()
        return content

    def read_item(self, path:str, type:str):
        content_type = "txt"
        if type in ["doc", "docx", "pdf"]:
            from asdTools.Classes.File.DocBase import DocBase
            content = DocBase(log_dir=self._log_dir).read_docxs(path)
        elif type in ["png", "jpg", "jpeg"]:
            from asdTools.Classes.API.EcloudAPI import EcloudAPI
            content = EcloudAPI(log_dir=self._log_dir).convert_img_to_str_by_OCR(path)
        elif type in ["txt"]:
            content = self.read_txt(path)
        elif type in ["json"]:
            content = self.read_json(path)
            content_type = "json"
        elif type in ["csv"]:
            from asdTools.Classes.Base.BaseModel import BaseModel
            content = BaseModel(log_dir=self._log_dir).read_csvLike(path)
        else:
            content = ""
        return content, content_type

    def read(self, file:dict, read_fn=None, save_fn=None, load_fn=None):
        if read_fn == None:
            read_fn = self.read_item
        if save_fn == None:
            save_fn = self.save_item
        if load_fn == None:
            load_fn = self.load_item
        content = file["content"]
        content_path = self.generate_storage_path(file, 0)
        if content == "":
            content, type = read_fn(file["path"], file["type"]["file"])
            file["content"] = content
            file["type"]["content"] = type
            save_fn(content, content_path, type)
            self.save_config()
        else:
            load_fn(content_path, file["type"]["content"])
        return content

    def recovery_config(self, read_fn=None):
        if read_fn == None:
            read_fn = self.read_item
        for id, item in self.config["files"].items():
            content_path = self.generate_storage_path(item, 0)
            if self.exists(content_path):
                content_type = item["type"]["content"]
                content, _ = read_fn(content_path, content_type)
                item["content"] = content
            analysis_path = self.generate_storage_path(item, 1)
            if self.exists(analysis_path):
                analysis_type = item["type"]["analysis"]
                analysis, _ = read_fn(analysis_path, analysis_type)
                item["analysis"] = analysis

    def save_config(self):
        _config = self.deepcopy(self.config)
        for item in _config["files"].keys():
            _config["files"][item]["content"] = ""
            _config["files"][item]["analysis"] = ""
        config_path = self.generate_path(output_dir=self._log_dir, output_file=self.config_file)
        self.save_file(_config, config_path)

    def sort(self, new_order:list):
        self.config["order"] = new_order
        self.counter.update(self.config["files"], self.config["order"], 0)
        self.save_config()

    def save_item(self, content, path:str, type:str):
        if type in ["json", "txt"]:
            self.save_file(content, path)
        else:
            self.save_file(content, path)

    def update(self, item:dict, save_fn=None):
        if not save_fn:
            save_fn = self.save_item
        save_path = self.generate_storage_path(item, 0)
        save_fn(item["content"], save_path, item["type"]["content"])
        save_path = self.generate_storage_path(item, 1)
        save_fn(item["analysis"], save_path, item["type"]["analysis"])
        
    