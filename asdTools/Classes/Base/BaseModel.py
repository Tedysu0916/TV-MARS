from asdTools.Classes.Base.CommandBase import CommandBase
from asdTools.Classes.Base.FormatBase import FormatBase
from asdTools.Classes.Base.SystemBase import SystemBase
from asdTools.Classes.Base.UnitBase import UnitBase
from asdTools.Classes.Base.TimeBase import TimeBase
from asdTools.Classes.Base.TypeBase import TypeBase
from asdTools.Classes.Base.VarBase import VarBase
from asdTools.Classes.Base.IOBase import IOBase


class BaseModel(CommandBase, FormatBase, SystemBase, UnitBase, TimeBase, TypeBase, VarBase, IOBase):
    def __init__(self, 
                 name:str="", 
                 log_dir:str="", 
                 log_file:str="", 
                 log_level:str="all", 
                 multipleFiles:str=False, 
                 log_info:tuple=None,
                 **kwargs) -> None:
        super(CommandBase, self).__init__(**kwargs)
        super(FormatBase, self).__init__(**kwargs)
        super(SystemBase, self).__init__(**kwargs)
        super(UnitBase, self).__init__(**kwargs)
        super(TimeBase, self).__init__(**kwargs)
        super(TypeBase, self).__init__(**kwargs)
        super(VarBase, self).__init__(**kwargs)
        super(IOBase, self).__init__(**kwargs)
        self._time_start = self.get_time(True)
        if name == "":
            self.name = self.__class__.__name__
        else:
            self.name = name
        # if multipleFiles:
        #     self._log_dir = f"./Logs/{self.name}/{self._time_start}" if log_dir == "" else log_dir
        #     self._log_dir_root = f"./Logs/{self.name}/{self._time_start}" if log_dir == "" else log_dir
        # else:
        self._log_dir = self.join("Logs", self.name) if log_dir == "" else log_dir
        self._log_dir_root = self._log_dir
        self._multipleFiles = multipleFiles
        if multipleFiles:
            self._log_dir = self.join(self._log_dir, self._time_start)

        self._log_file = f"{self.name}_{self._time_start}.log" if log_file == "" else log_file
        self._log_level = log_level
        self._log_level_table = {"message": 0, "all": 0, "warning": 1, "error": 2, "none":3}
        self._done_msg = []
        if log_info:
            self.set_log(log_info)

    def begining(self, message="", isSimple:bool=False, isZh:bool=False) -> None:
        if isSimple:
            if isZh: self.log(f"------ 日志保存至 {self.convert_path_to_abspath(self._log_dir)} ------")
            else:    self.log(f"------ Output are saved in {self.convert_path_to_abspath(self._log_dir)} ------")
        else:
            self.separator("Begining")
            if isZh:
                self.log(f"开始于{self._time_start}")
                self.log(f"输出文件保存至{self.convert_path_to_abspath(self._log_dir)}")
                self.log(f"日志信息保存至{self._log_file}")
            else:
                self.log(f"Start in {self._time_start}")
                self.log(f"Output files are saved in {self.convert_path_to_abspath(self._log_dir)}")
                self.log(f"Output log is saved as {self._log_file}")
        if message != "":
            if isinstance(message, str):
                self.log(message)
            elif isinstance(message, list):
                for msg in message:
                    self.log(msg)
            else:
                self.log(str(message))
        for msg in self._done_msg:
            self.log(msg)

    def done(self, message="", isSimple:bool=False, isZh:bool=False) -> None:
        self._time_end = self.get_time(True)
        if isSimple:
            if isZh: self.log(f"------ 日志保存至 {self.convert_path_to_abspath(self._log_dir)} ------")
            else:    self.log(f"------ Output are saved in {self.convert_path_to_abspath(self._log_dir)} ------")
        else:
            self.separator("Finished")
            if isZh:
                self.log(f"开始于 {self._time_start}, 结束于 {self._time_end}")
                self.log(f"输出文件保存至: {self.convert_path_to_abspath(self._log_dir)}")
                self.log(f"日志信息保存至: {self._log_file}")
            else:
                self.log(f"Done. Start in {self._time_start}, end in {self._time_end}")
                self.log(f"Output files are saved in {self.convert_path_to_abspath(self._log_dir)}")
                self.log(f"Output log is saved as {self._log_file}")
        if message != "":
            if isinstance(message, str):
                self.log(message)
            elif isinstance(message, list):
                for msg in message:
                    self.log(msg)
            else:
                self.log(str(message))
        for msg in self._done_msg:
            self.log(msg)

    def generate_output_path(self, output_dir:str="", output_middle_dir:str="", output_file:str="", createIfNotExists=True):
        if output_dir == "":
            output_dir = self._log_dir
        if output_file == "":
            output_file = f"{self.name}_{self.get_time(True)}.log"
        output_path = self.generate_path(
            output_dir=output_dir, 
            output_middle_dir=output_middle_dir, 
            output_file=output_file, 
            createIfNotExists=createIfNotExists)
        return output_path

    def get_log(self):
        return self._log_dir_root, self._log_file, self._time_start, self.name

    def log(self, content, logTime:bool=True, level:str="message", logWhenDone:bool=False) -> str:
        time_current = self.get_time()
        if logWhenDone:
            self._done_msg.append(content)
        if self._log_level_table[level] >= self._log_level_table[self._log_level]:
            if level == "warning":
                content = "Warning: " + content
            if level == "error":
                content = "Error: " + content
            if logTime:
                content = f"{time_current}: {content}"
            content_end = '' if content[-1] == '\n' else '\n'
            print(content, end=content_end)
            log_path = self.generate_output_path(output_file=self._log_file)
            self.save_file(f"{content}{content_end}", log_path, mode='a')
        return time_current

    def input(self, message="Input: ", needLog=False) -> str:
        """
        Displays a message and returns the user input as a string.

        Args:
            message (str): Message to display. Defaults to "Input: ".

        Returns:
            str: User input as a string.
        """
        content = input(message)
        if needLog:
            self.log(f"{message}{content}")
        return content

    def raise_error(self, message:str, statue_code:int=-1) -> None:
        """
        Logs an error message and exits the program with a status code.

        Args:
            message (str): Error message to log.
            statue_code (int): Status code to exit with. Defaults to -1.
        """
        self.log(message, level="error")
        self.exit(statue_code)

    def separator(self, message:str) -> None:
        self.log(f"")
        self.log(f"---------- {message} ----------")

    def set_log(self, log_info:tuple):
        self._log_dir_root = log_info[0]
        self._log_file = log_info[1]
        self._time_start = log_info[2]
        self.name = log_info[3]
        if self._multipleFiles:
            self._log_dir = self.join(self._log_dir_root, self._time_start)
        else:
            self._log_dir = self._log_dir_root

    def warning(self, message:str) -> None:
        self.log(message, level="warning")