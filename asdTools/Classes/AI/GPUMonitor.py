from asdTools.Classes.Base.BaseModel import BaseModel
import pynvml
import random


class GPUMonitor(BaseModel):
    """
    Monitor available GPUs on NVIDIA GPUs.
    """
    def __init__(self, 
                 capacity_thres:float=0, 
                 output_type:str="list", 
                 time_cooldown:int=60,
                 gpu_log_file:str='GPUMonitor.json',
                 **kwargs) -> None:
        """Initialize GPUMonitor object.

        Args:
            capacity_thres (float): The threshold of available GPU memory capacity (in GB) required for a GPU to be considered as available. Default is 0.
            output_type (str): The format of output. Available options are "list" and "str". Default is "list".
            time_cooldown (int): The time (in seconds) after which a GPU is considered as spare. Default is 300.
            gpu_log_file (str): The file name to store the GPU information. Default is 'GPUMonitor.json'.
            **kwargs: Additional keyword arguments.

        Examples:
        If you need 2 GPUs, and each gpus should have 8G memory at least:
        monitor = GPUMonitor(capacity_thres=8, output_type="list", time_cooldown=10)
        gpus_available = monitor(gpus_id=[0, 1, 2, 3], gpu_need=2, extra_info="test")
        --> 2023-05-14 20:14:06: test_No available gpus, sleep 3.99s.
        --> 2023-05-14 20:14:10: test_No available gpus, sleep 2.51s.
        --> 2023-05-14 20:14:13: test_Find available gpus: [0, 1].
        --> output: [0, 1]
        
        If you need 4 GPUs, and each gpus should have 18G memory at least:
        gpus_available = GPUMonitor()(gpus_id=[0, 1, 2, 3, 4, 5, 6, 7], gpu_need=4, capacity_thres=18, output_type="str", time_cooldown=300)
        --> output: "0,1,2,3"  

        """
        super().__init__(**kwargs)
        self.capacity_thres = capacity_thres
        self.output_type = output_type
        self.time_cooldown = time_cooldown
        self.gpu_log_path = self.generate_output_path(output_file=gpu_log_file)
        if not self.exists(self.gpu_log_path):
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            self.gpu_info = {}
            for i in range(self.device_count):
                self.gpu_info[str(i)] = self.get_time(timestamp=0)
            self.save_file(self.gpu_info, self.gpu_log_path)
        else:
            self.gpu_info = self.read_json(self.gpu_log_path)

    def __call__(self, 
                 gpus_id:list, 
                 gpu_need:int, 
                 capacity_thres:float=-1, 
                 output_type:str="", 
                 time_cooldown:int=-1,
                 extra_info:str=""):
        """Call GPUMonitor object to get available GPUs.

        Args:
            gpus_id (list): The list of GPU IDs to search from.
            gpu_need (int): The number of GPUs required.
            capacity_thres (float): The threshold of available GPU memory capacity (in GB) required for a GPU to be considered as available. If not provided, use the value set in the initialization.
            output_type (str): The format of output. Available options are "list" and "str". If not provided, use the value set in the initialization.
            time_cooldown (int): The time (in seconds) after which a GPU is considered as spare. If not provided, use the value set in the initialization.
            extra_info (str): Additional information to be logged. Default is an empty string.

        Returns:
            The list of available GPUs.
        """
        gpus_id = self.get_available_gpus(gpus_id, gpu_need, capacity_thres, output_type, time_cooldown, extra_info)
        return gpus_id

    def get_available_gpus(self, 
                           gpus_id:list, 
                           gpu_need:int, 
                           capacity_thres:float=-1, 
                           output_type:str="", 
                           time_cooldown:int=-1, 
                           extra_info:str=""):
        """
        Finds available GPUs based on the specified parameters.

        Args:
        - gpus_id (list): A list of GPU IDs to search for available GPUs.
        - gpu_need (int): The number of available GPUs needed.
        - capacity_thres (float): The minimum empty memory capacity required for a GPU.
                                  Defaults to the value of self.capacity_thres if not specified.
        - output_type (str): The format of the output. Can be "list" or "str". Defaults to "" if not specified.
        - time_cooldown (int): The minimum time that a GPU must have been idle to be considered available, in seconds.
                               Defaults to the value of self.time_cooldown if not specified.
        - extra_info (str): Additional information to be logged. Defaults to "" if not specified.

        Returns:
        - result (list or str): A list or string of available GPUs, depending on the value of output_type.

        Raises:
        - ValueError: If an invalid value is passed for output_type.
        """
        if capacity_thres == -1:
            capacity_thres = self.capacity_thres
        if output_type == "":
            output_type = self.output_type
        if time_cooldown == -1:
            time_cooldown = self.time_cooldown
        while True:
            gpus_available = self.get_empty_gpus(gpus_id, capacity_thres)
            gpus_available = self.get_spare_gpus(gpus_available, time_cooldown)
            if len(gpus_available) >= gpu_need:
                gpus_available = gpus_available[:gpu_need]
                time_now = self.get_time()
                for gpu_id in gpus_available:
                    self.gpu_info[str(gpu_id)] = time_now
                self.save_file(self.gpu_info, self.gpu_log_path)
                if output_type == "list":
                    result = gpus_available
                elif output_type == "str":
                    try:
                        result = ",".join(gpus_available)
                    except:
                        result = str(gpus_available[0])
                else:
                    self.raise_error(f"No vaild output type: {output_type}.")
                self.log(f"{extra_info}_Find available gpus: {str(result)}.")
                return result
            else:
                time_sleep = round(random.uniform(1, time_cooldown), 2)
                self.log(f"{extra_info}_No available gpus, sleep {time_sleep}s.")
                self.sleep(time_sleep)

    def get_empty_gpus(self, gpus_id:list, capacity:float) -> list:
        """
        Get the list of GPUs with free memory capacity greater than the specified capacity.

        Args:
        - gpus_id (list): List of GPU IDs to be checked for free memory capacity.
        - capacity (float): Minimum amount of free memory capacity required for the GPU to be considered available.

        Returns:
        - gpus_available (list): List of GPU IDs with free memory capacity greater than the specified capacity.
        """
        gpus_free = self.get_gpus_free(gpus_id)
        gpus_available = []
        for i, gpu_free in enumerate(gpus_free):
            if gpu_free > capacity:
                gpus_available.append(gpus_id[i])
        return gpus_available

    def get_spare_gpus(self, gpus_id:list, time_cooldown:int=60) -> list:
        """
        To prevent simultaneously occupy the GPUs.
        Get the list of GPUs that have been idle for longer than the specified time.

        Args:
        - gpus_id (list): List of GPU IDs to be checked for idle time.
        - time_cooldown (int): Minimum time in seconds for the GPU to be considered available.

        Returns:
        - gpus_available (list): List of GPU IDs that have been idle for longer than the specified time.
        """
        gpus_info = self.read_json(self.gpu_log_path)
        gpus_available = []
        time_now = self.get_time()
        for gpu_id in gpus_id:
            time_gpu = gpus_info[str(gpu_id)]
            time_diff = self.get_time_diff(time_gpu, time_now)
            if time_diff >= time_cooldown:
                gpus_available.append(gpu_id)
        return gpus_available
            
    @staticmethod
    def get_gpu_free(gpu_id:int=0) -> int:
        """
        Get the amount of free memory available in the specified GPU.

        Args:
        - gpu_id (int): ID of the GPU to get free memory capacity.

        Returns:
        - free (int): Amount of free memory in the specified GPU in bytes.
        """
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free = gpu_memory_info.free
        pynvml.nvmlShutdown()
        return free

    def get_gpus_free(self, gpus_id:list=[0]) -> list:
        """
        Get the list of free memory capacities of the specified GPUs in gigabytes.

        Args:
        - gpus_id (list): List of GPU IDs to get free memory capacities.

        Returns:
        - gpus_free (list): List of free memory capacities of the specified GPUs in gigabytes.
        """
        gpus_free = []
        for gpu_id in gpus_id:
            gpu_free = self.get_gpu_free(gpu_id)
            gpu_free = self.convert_storage_units(gpu_free, "B", "GB")
            gpus_free.append(gpu_free)
        return gpus_free

if __name__ == "__main__":
    monitor = GPUMonitor(capacity_thres=4, output_type="list", time_cooldown=60)(gpus_id=[0, 1, 2, 3], gpu_need=2)
    monitor([0], 1, 1, output_type="list", time_cooldown=10, extra_info="test")

