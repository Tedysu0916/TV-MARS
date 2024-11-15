from asdTools.Classes.API.LLMsAPI import LLMsAPI


if __name__ == "__main__":
    """
    用于和LLMs聊天，示例为与ChatGLM本地部署的API通信。
    Used for chatting with LLMs, with an example of communicating with the locally deployed ChatGLM API.
    """
    url = "http://localhost:8000"
    print("Support command:")
    print("\exit: Terminate chating.")
    ChatBot = LLMsAPI(url)
    ChatBot.chating()