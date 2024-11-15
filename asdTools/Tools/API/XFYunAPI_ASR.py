# -*- coding: utf-8 -*-
from asdTools.Secret.key import XFYunAPPID, XFYunAPISecretASR
from asdTools.Classes.Base.BaseModel import BaseModel
import requests
import hashlib
import urllib
import base64
import hmac
import json
import time
import os

lfasr_host = 'https://raasr.xfyun.cn/v2/api'
# 请求的接口名
api_upload = '/upload'
api_get_result = '/getResult'


class RequestApi(object):
    def __init__(self, appid, secret_key, upload_file_path):
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path
        self.ts = str(int(time.time()))
        self.signa = self.get_signa()

    def get_signa(self):
        appid = self.appid
        secret_key = self.secret_key
        m2 = hashlib.md5()
        m2.update((appid + self.ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        # 以secret_key为key, 上面的md5为msg， 使用hashlib.sha1加密结果为signa
        signa = hmac.new(secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        return signa


    def upload(self):
        print("上传部分：")
        upload_file_path = self.upload_file_path
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)

        param_dict = {}
        param_dict['appId'] = self.appid
        param_dict['signa'] = self.signa
        param_dict['ts'] = self.ts
        param_dict["fileSize"] = file_len
        param_dict["fileName"] = file_name
        param_dict["duration"] = "200"
        # print("upload参数：", param_dict)
        data = open(upload_file_path, 'rb').read(file_len)

        response = requests.post(url =lfasr_host + api_upload+"?"+urllib.parse.urlencode(param_dict),
                                headers = {"Content-type":"application/json"},data=data)
        # print("upload_url:",response.request.url)
        result = json.loads(response.text)
        # print("upload resp:", result)
        return result


    def get_result(self):
        uploadresp = self.upload()
        orderId = uploadresp['content']['orderId']
        param_dict = {}
        param_dict['appId'] = self.appid
        param_dict['signa'] = self.signa
        param_dict['ts'] = self.ts
        param_dict['orderId'] = orderId
        param_dict['resultType'] = "transfer,predict"
        # print("")
        # print("查询部分：")
        # print("get result参数：", param_dict)
        status = 3
        # 建议使用回调的方式查询结果，查询接口有请求频率限制
        while status == 3:
            response = requests.post(url=lfasr_host + api_get_result + "?" + urllib.parse.urlencode(param_dict),
                                     headers={"Content-type": "application/json"})
            # print("get_result_url:",response.request.url)
            result = json.loads(response.text)
            # print(result)
            status = result['content']['orderInfo']['status']
            # print("status=",status)
            if status == 4:
                break
            time.sleep(5)
        # print("get_result resp:",result)
        return result

def convert_audio_to_txt(audio_file):
    api = RequestApi(appid=XFYunAPPID,
                     secret_key=XFYunAPISecretASR,
                     upload_file_path=audio_file)
    json_recover = BaseModel()
    response = api.get_result()
    info = json.loads(response["content"]["orderResult"])
    lattice_data = json_recover.convert_val_adaptive(info["lattice"])
    text = ""
    for sentence in lattice_data:
        for word in sentence["json_1best"]["st"]["rt"][0]["ws"]:
            text += word["cw"][0]["w"]
    return text


# 输入讯飞开放平台的appid，secret_key和待转写的文件路径
if __name__ == '__main__':
    api = RequestApi(appid=XFYunAPPID,
                     secret_key=XFYunAPISecretASR,
                     upload_file_path=r"F:\0_DATA\1_DATA\CODE\PYTHON\202304_RJB_C4\other\audio\lfasr_涉政.wav")
    json_recover = BaseModel()
    response = api.get_result()
    info = json.loads(response["content"]["orderResult"])
    lattice_data = json_recover.convert_val_adaptive(info["lattice"])
    text = ""
    for sentence in lattice_data:
        for word in sentence["json_1best"]["st"]["rt"][0]["ws"]:
            text += word["cw"][0]["w"]
    # text = "".join([item['ws'][0]['cw'][0]['w'] for item in lattice_data])
    0
