from asdTools.Classes.Base.RewriteBase import RewriteBase
import sys
import os


class CommandBase(RewriteBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def check_command(command:str) -> bool:
        """
        Check if the given command is a valid command.

        Args:
        - command: A string representing the command to be checked.

        Returns:
        - A boolean indicating whether the command is valid or not.
        """
        if command[0] == '\\':
            if command[1] == '\\':
                return False
        return False

    def exec_command(self, command:str) -> None:
        """
        Execute the given command.

        Args:
        - command: A string representing the command to be executed.

        Returns:
        - None
        """
        command = command[1:]
        if command == "exit":
            self.exit(0)

    @staticmethod
    def exit(statue_code:int=0) -> None:
        """
        Exit the program with the given exit code.

        Args:
        - statue_code: An integer representing the exit code.

        Returns:
        - None
        """
        sys.exit(statue_code)

    @staticmethod
    def pause() -> None:
        """
        Pause the program execution until the user presses any key.

        Args:
        - None

        Returns:
        - None
        """
        os.system("pause")