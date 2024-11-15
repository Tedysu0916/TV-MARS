from asdTools.Classes.Base.RewriteBase import RewriteBase
import platform
import socket


class SystemBase(RewriteBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def check_Windows(self) -> bool:
        """
        Check if the current operating system is Windows.

        Returns:
            bool: True if the current OS is Windows, False otherwise.
        """
        plat = platform.system().lower()
        if plat == 'windows':
            return True
        else:
            return False

    def check_Linux(self) -> bool:
        """
        Check if the current operating system is Linux.

        Returns:
            bool: True if the current OS is Linux, False otherwise.
        """
        plat = platform.system().lower()
        if plat == 'linux':
            return True
        else:
            return False

    def get_hostname(self):
        """
        Get the hostname of the system.

        Returns:
            str: The hostname of the system.
        """
        hostname = socket.gethostname()
        return hostname

    def isAuthor(self):
        hostname = self.get_hostname()
        if hostname == "MWHLS":
            return True
        else:
            return False