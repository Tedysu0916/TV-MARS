from asdTools.Classes.Base.RewriteBase import RewriteBase
import hashlib
import shutil
import json
import os


class IOBase(RewriteBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def add_suffix(self, path:str, suffix:str="_suffix") -> str:
        """
        Add suffix to the directory or file name in the given path and return the new path.

        Args:
        - path (str): Path to the directory or file.
        - suffix (str, optional): Suffix to add to the file or directory name. Default is "_suffix".

        Returns:
        - str: Path with added suffix.

        Raises:
        - None

        Example:
        >>> add_suffix('/path/to/directory', '_new')
        '/path/to/directory_new'
        >>> add_suffix('/path/to/file.txt', '_new')
        '/path/to/file_new.txt'
        """
        if os.path.isdir(path):
            dir_root, dir_name = os.path.split(path)
            dir_name += suffix
            path = os.path.join(dir_root, dir_name)
        elif os.path.isfile(path):
            file_root, file_name = os.path.split(path)
            file_name_root, file_name_ext = os.path.splitext(file_name)
            file_name = file_name_root + suffix + file_name_ext
            path = os.path.join(file_root, file_name)
        else:
            path += suffix
        return path

    def convert_path_to_abspath(self, path:str) -> str:
        abspath = os.path.abspath(path)
        return abspath

    def get_paths_from_dir(self, path:str, 
                          type:str="file", 
                          needAbsPath:bool=False, 
                          keepExt:bool=True, 
                          includeSubfolder:bool=True)->list:
        """Return a list of file or directory paths in the specified directory and its subfolders.

        Args:
            path (str): The path to the directory to search.
            type (str, optional): The type of paths to include, either "file", "dir", or "filedir" (both). Defaults to "file".
            needAbsPath (bool, optional): If True, return absolute paths. Defaults to False.
            keepExt (bool, optional): If True, keep the file extension in the path. Defaults to True.
            includeSubfolder (bool, optional): If True, search in subfolders recursively. Defaults to True.

        Returns:
            list: A list of paths of the specified type found in the directory and its subfolders.

        """
        if os.path.isfile(path):
            return [path]
        paths = []
        for root, dirs, files in os.walk(path):
            if "file" in type:
                for file in files:
                    file_path = os.path.join(root, file)
                    if needAbsPath:
                        file_path = os.path.abspath(file_path)
                    if not keepExt:
                        file_path, _ = os.path.splitext(file_path)
                    paths.append(file_path)
            if "dir" in type:
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    paths.append(dir_path)
            if not includeSubfolder:
                break
        return paths

    def get_dir_of_file(self, path:str, lastDir:bool=False, root:str="") -> str:
        dir_name = os.path.dirname(path)
        if lastDir:
            dir_name = os.path.basename(dir_name)
        elif root != "":
            dir_name = dir_name[len(root)+1:]
        return dir_name
    
    def get_hierarchy_of_file(self, path:str) -> list:
        _path = path
        hierarchy = []
        while True:
            _path, folder = os.path.split(_path)
            if folder:
                hierarchy.insert(0, folder)
            else:
                if _path:
                    hierarchy.insert(0, _path)
                break
        return hierarchy

    def get_name_of_file(self, path:str, keepExt:bool=False) -> str:
        """
        Returns the name of the given file path.

        Args:
        - path (str): The file path.
        - keepExt (bool): If True, returns the file name along with its extension.

        Returns:
        - str: The name of the given file path.

        """
        _, file_name = os.path.split(path)
        if keepExt:
            return file_name
        file, _ = os.path.splitext(file_name)
        return file

    def get_ext_of_file(self, path:str) -> str:
        """
        Returns the extension of the given file path.

        Args:
        - path (str): The file path.

        Returns:
        - str: The extension of the given file path.

        """
        _, file_name = os.path.split(path)
        _, ext = os.path.splitext(file_name)
        return ext[1:]

    def get_loggingLogger_path(self, logger, needDir:bool=True) -> str:
        import logging
        handlers = logger.handlers
        for handler in handlers:
            if isinstance(handler, logging.FileHandler):
                return self.get_dir_of_file(handler.baseFilename) if needDir else handler.baseFilename
        logger = logger.parent
        return self.get_loggingLogger_path(logger, needDir)

    def get_name_of_files(self, files:list, keepExt:bool=False) -> list:
        res = []
        for file in files:
            file_name = self.get_name_of_file(file, keepExt)
            res.append(file_name)
        return res

    def get_md5_of_file(self, path:str) -> str:
        with open(path, 'rb') as f:
            md5_hash = hashlib.md5()
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                md5_hash.update(chunk)
        md5_digest = md5_hash.hexdigest()
        return md5_digest

    def get_md5_of_txt(self, txt:str) -> str:
        md5_hash = hashlib.md5()
        md5_hash.update(txt.encode('utf-8'))
        md5_digest = md5_hash.hexdigest()
        return md5_digest
    
    def get_size_of_file(self, path:str, unit:str="MB"):
        size = os.path.getsize(path)
        size = self.convert_storage_units(size, "B", unit)
        return size
    
    def get_size_of_files(self, files:list, unit:str="MB"):
        sizes = []
        for file in files:
            size = self.get_size_of_file(file, unit)
            sizes.append(size)
        return sizes

    def generate_path(self, output_dir:str="", output_middle_dir:str="", output_file:str="", createIfNotExists=True):
        output_dir = self.join(output_dir, output_middle_dir)
        output_path = self.join(output_dir, output_file)
        if createIfNotExists:
            self.mkdir(output_dir)
        return output_path

    def join(self, *arg:str) -> str:
        """
        Join one or more path components, regardless of operating system, and return the combined path.

        Args:
            *arg (str): one or more path components to join.

        Returns:
            str: the combined path.

        Example:
            >>> FileUtil.join_path('path', 'to', 'file.txt')
            'path/to/file.txt'
        """
        return os.path.join(*arg)
    
    def mkdir(self, path:str) -> bool:
        if not os.path.exists(path):
            os.makedirs(path)
            return True
        return False

    def move(self, src:str, dest:str):
        shutil.move(src, dest)

    def copy(self, src:str, dest:str):
        dir_dest = self.get_dir_of_file(dest)
        self.mkdir(dir_dest)
        shutil.copy(src, dest)

    def exists(self, path:str):
        isExists = os.path.exists(path)
        return isExists

    def save_file(self, 
                  content, 
                  log_path:str="./Logs/log.txt",
                  mode:str='w') -> str:
        log_dir = self.get_dir_of_file(log_path)
        self.mkdir(log_dir)
        with open(log_path, mode, encoding='utf8') as f:
            try:
                if isinstance(content, list):
                    f.write("".join(content))
                elif isinstance(content, str):
                    f.write(content)
                elif isinstance(content, dict):
                    json.dump(content, f, indent=2, sort_keys=True, ensure_ascii=False)
                else:
                    content_new = str(content)
                    f.write(content_new)
            except:
                content_new = str(content)
                f.write(content_new)
                self.warning(f"Did not match to the appropriate type, forced to write to the file in {log_path}, please check")
        return log_path

    def read_json(self, path:str) -> dict:
        with open(path, "r", encoding="utf8") as f:
            content = json.load(f)
        return content

    def read_txt(self, path:str, asLine:bool=True) :
        with open(path, "r", encoding="utf8") as f:
            content = f.readlines()
            if asLine:
                content = "".join(content)
            else:
                # remove "\n"
                content = [x.rstrip("\n") for x in content]
        return content

    def read_html(self, path:str) -> str:
        with open(path, "r", encoding="utf8") as f:
            content = f.read()
        return content

    def remove(self, path:str) -> str:
        os.remove(path)
        return path
    
    def remove_root_of_path(self, path:str, root:str) -> str:
        root = os.path.join(root, "remove_root_of_path.asdTools.tmp")
        root = root.replace("remove_root_of_path.asdTools.tmp", "")
        path = path.replace(root, "")
        return path

    def check_file(self, path:str) -> bool:
        """
        Checks if the given path points to a file.

        Args:
        - path (str): The path to check.

        Returns:
        - bool: True if the path points to a file, False otherwise.

        """
        if os.path.isfile(path):
            return True
        return False
        
    def check_dir(self, path:str) -> bool:
        """
        Checks if the given path points to a directory.

        Args:
        - path (str): The path to check.

        Returns:
        - bool: True if the path points to a directory, False otherwise.

        """
        if os.path.isdir(path):
            return True
        return False

    def check_path(self, path:str) -> str:
        """
        Checks if the given path points to a valid location.

        Args:
        - path (str): The path to check.

        Returns:
        - str: False if the path points to a file or directory, True otherwise.

        """
        if os.path.isfile(path):
            return True
        elif os.path.isdir(path):
            return True
        else:
            return False

    def check_ext(self, path:str, ext_list:list, allow:bool=True) -> bool:
        """
        Check whether a given path has a valid extension.

        Args:
        - path (str): The path to be checked.
        - ext_list (list): A list of valid extensions.
        - allow (bool): A boolean flag indicating whether the extension should be allowed or not. Default is True.

        Returns:
        - bool: True if the path has a valid extension and allow is True, or if the path does not have a valid extension and allow is False. False otherwise.

        """
        # v0.0.14: add support for ".suffix".
        _, file_name = os.path.split(path)
        _, ext = os.path.splitext(file_name)
        # ext = ext[1:]
        if ext in ext_list:
            return True if allow else False
        elif ext[1:] in ext_list:
            return True if allow else False
        else:
            return False if allow else True

    def check_name(self, path:str, name_list:list, keepExt:bool=True, allow:bool=True) -> bool:
        """
        Check whether a given path has a valid file name.

        Args:
        - path (str): The path to be checked.
        - name_list (list): A list of valid file names.
        - keepExt (bool): A boolean flag indicating whether to keep the file extension. Default is True.
        - allow (bool): A boolean flag indicating whether the file name should be allowed or not. Default is True.

        Returns:
        - bool: True if the path has a valid file name and allow is True, or if the path does not have a valid file name and allow is False. False otherwise.

        """
        _, file_name = os.path.split(path)
        if not keepExt:
            file_name, _ = os.path.splitext(file_name)
        if file_name in name_list:
            return True if allow else False
        else:
            return False if allow else True
        
    def check_size(self, path:str, max_size:float, unit:str="MB", allow:bool=True):
        size = self.get_size_of_file(path, unit)
        if size < max_size:
            return True if allow else False
        else:
            return False if allow else True

    def filter_ext(self, paths:list, ext_list:list=[], allow:bool=True) -> list:
        """
        Filters a list of file paths based on the extensions.

        Args:
        - paths (list): A list of file paths to filter.
        - ext_list (list): A list of extensions to allow or disallow. If empty, all extensions will be allowed.
        - allow (bool): If True, the files with the extensions in the ext_list will be allowed. If False, they will be disallowed.

        Returns:
        - list: A list of file paths filtered based on the given extensions.

        """
        res = []
        for path in paths:
            if self.check_ext(path, ext_list, allow):
                res.append(path)
        return res
    
    def filter_size(self, files:list, max_size:float, unit:str="MB", allow:bool=True):
        res = []
        for file in files:
            if self.check_size(file, max_size, unit, allow):
                res.append(file)
        return res
    
    def sort_files_size(self, files:list, reverse:bool=False) -> list:
        """
        Sorts a list of file paths based on their sizes.

        Args:
        - files (list): A list of file paths to sort.
        - reverse (bool): A boolean flag indicating whether to sort in descending order. Default is False.

        Returns:
        - list: A sorted list of file paths based on their sizes.

        """
        sort_files = sorted(files, key=lambda x: os.path.getsize(x), reverse=reverse)
        return sort_files