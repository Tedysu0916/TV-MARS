from asdTools.Classes.AI.GPUMonitor import GPUMonitor


def SampleInChinese():
    """
    假设我有8张卡，我想获取4张卡，每张卡至少16G显存，并且我需要字符串格式的结果。
    使用如下代码，将持续搜索可用GPU，每次搜索间隔不超过300秒。
    搜索不到时，等待random(1, 300)秒后，再次搜索。
    如果某次搜索发现 GPU 0, 1, 4, 5, 6, 7 空闲，则会返回"0,1,4,5"，并结束搜索。
    """
    gpus_id = [0, 1, 2, 3, 4, 5, 6, 7]
    # 方式1
    monitor = GPUMonitor(capacity_thres=16, output_type="str", time_cooldown=300)
    gpus_available = monitor(gpus_id, gpu_need=4)
    # 方式2
    gpus_available = GPUMonitor()(gpus_id, gpu_need=4, capacity_thres=16, output_type="str", time_cooldown=300)

    """
    我想知道哪些实验正在等待GPU，则使用extra_info，为每次输出增加前缀。
    我需要list类型的结果，则output_type="list"
    2023-05-14 20:14:06: test_No available gpus, sleep 3.99s.
    2023-05-14 20:14:10: test_No available gpus, sleep 2.51s.
    2023-05-14 20:14:13: test_Find available gpus: [0, 1, 4, 5].
    """
    gpus_available = monitor(gpus_id, gpu_need=4, extra_info="test", output_type="list")
    
def SampleInEnglish():
    """
    Assume that there are 8 GPUs available and I want to acquire 4 GPUs with at least 16GB memory per GPU. I need the result in string format.
    To achieve this, use the following code to continuously search for available GPUs with a search interval of no more than 300 seconds.
    If no available GPUs are found, wait for random(1, 300) seconds before searching again.
    If during a search, GPUs 0, 1, 4, 5, 6, 7 are found to be available, "0,1,4,5" will be returned and the search will end.
    """
    gpus_id = [0, 1, 2, 3, 4, 5, 6, 7]
    # Method 1
    monitor = GPUMonitor(capacity_thres=16, output_type="str", time_cooldown=300)
    gpus_available = monitor(gpus_id, gpu_need=4)
    # Method 2
    gpus_available = GPUMonitor()(gpus_id, gpu_need=4, capacity_thres=16, output_type="str", time_cooldown=300)

    """
    If I want to know which experiments are waiting for GPUs, I can use extra_info to add a prefix to each output.
    I need the result in list format, so output_type="list".
    Example Output:
    2023-05-14 20:14:06: test_No available gpus, sleep 3.99s.
    2023-05-14 20:14:10: test_No available gpus, sleep 2.51s.
    2023-05-14 20:14:13: test_Find available gpus: [0, 1, 4, 5].
    """
    gpus_available = monitor(gpus_id, gpu_need=4, extra_info="test", output_type="list")


if __name__ == "__main__":
    SampleInChinese()
    SampleInEnglish()