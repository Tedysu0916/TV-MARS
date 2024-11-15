import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket
from asdTools.Secret.key import XFYunAPIKeySpark, XFYunAPISecretSpark, XFYunAPPID

class Ws_Param(object):
    # 初始化
    def __init__(self):
        self.APPID = XFYunAPPID
        self.APIKey = XFYunAPIKeySpark
        self.APISecret = XFYunAPISecretSpark
        self.gpt_url = "ws://spark-api.xf-yun.com/v1.1/chat"
        self.host = urlparse(self.gpt_url).netloc
        self.path = urlparse(self.gpt_url).path

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.gpt_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url


# 收到websocket错误的处理
def on_error(ws, error, *args):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws):
    pass
    # print("### closed ###")


# 收到websocket连接建立的处理
def on_open(ws):
    thread.start_new_thread(run, (ws,))


def run(ws, *args):
    data = json.dumps(gen_params(
        appid=ws.appid, 
        question=ws.question, 
        history=ws.history,
        max_length=ws.max_length,
        temperature=ws.temperature))
    ws.send(data)


# 收到websocket消息的处理
def on_message(ws, message):
    # print(message)
    data = json.loads(message)
    code = data['header']['code']
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        # print(content, end='')
        ws.response += content
        if status == 2:
            ws.history.append([ws.question, ws.response])
            ws.result =  ws.response
            ws.response = ""
            ws.close()


def gen_params(appid, question, history=None, max_length=2048, temperature=0.01):
    """
    通过appid和用户的提问来生成请参数
    """
    data = {
        "header": {
            "app_id": appid,
            "uid": "1234"
        },
        "parameter": {
            "chat": {
                "domain": "general",
                "random_threshold": 0.5,
                "max_tokens": max_length,
                "auditing": "default",
                "temperature": temperature,
            }
        },
        "payload": {
            "message": {
                "text": []
            }
        }
    }
    for his in history:
        data["payload"]["message"]["text"].append({"role": "user", "content": his[0]})
        data["payload"]["message"]["text"].append({"role": "assistant", "content": his[1]})
    data["payload"]["message"]["text"].append({"role": "user", "content": question})
    return data


def call_spark(question, history=[], max_length=2048, temperture=0.01):
    wsParam = Ws_Param()
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = XFYunAPPID
    ws.history = history
    ws.max_length = max_length
    ws.temperture = temperture
    ws.response = ""
    ws.question = question
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    return ws.result, ws.history


if __name__ == "__main__":
    # 测试时候在此处正确填写相关信息即可运行
    result, history = call_spark("请回答‘你好’")
    result, history = call_spark("请回答'我在'", history)
    print(result)