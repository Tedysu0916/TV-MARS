from asdTools.Classes.Base.RewriteBase import RewriteBase
from datetime import datetime
import time


class TimeBase(RewriteBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_time(self, forFile:bool=False, timestamp:float=-1) -> str:
        """
        Get the current time.

        Args:
            forFile (bool): Specify if the time format should be suitable for file names.
            timestamp (float): Specify a timestamp to convert to time. Default is -1, which represents the current time.

        Returns:
            str: The current time in the specified format.
        """
        if timestamp == -1:
            time = datetime.now()
        else:
            time = datetime.fromtimestamp(0)
        if forFile:
            time = time.strftime(f"%Y-%m-%d_%H-%M-%S")
        else:
            time = time.strftime(f"%Y-%m-%d %H:%M:%S")
        return time

    def get_timestamp(self, time:datetime=None) -> float:
        """
        Get the timestamp from a given time.

        Args:
            time (datetime): The time to convert to timestamp. Default is None, which represents the current time.

        Returns:
            float: The timestamp corresponding to the given time.
        """
        if time == None:
            time = datetime.now()
        timestamp = time.timestamp()
        return timestamp

    def get_time_diff(self, time1:str, time2:str) -> float:
        """
        Calculate the time difference between two time values.

        Args:
            time1 (str): The first time value.
            time2 (str): The second time value.

        Returns:
            float: The time difference in seconds.

        Raises:
            ValueError: If an invalid time format is provided.
        """
        time_formats = ("%Y-%m-%d_%H-%M-%S", 
                        "%Y-%m-%d %H:%M:%S")
        for time_format in time_formats:
            try:
                time1 = datetime.strptime(time1, time_format)
                time2 = datetime.strptime(time2, time_format)
                time_diff = time2 - time1
                time_diff = time_diff.total_seconds()
                return time_diff
            except ValueError:
                pass
        raise ValueError("Invalid time format")

    def sleep(self, time_sleep):
        """
        Pause the execution for a specified amount of time.

        Args:
            time_sleep: The time to sleep in seconds.
        """
        time.sleep(time_sleep)