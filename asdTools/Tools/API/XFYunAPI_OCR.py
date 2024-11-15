
"""
  印刷文字识别WebAPI接口调用示例接口文档(必看)：https://doc.xfyun.cn/rest_api/%E5%8D%B0%E5%88%B7%E6%96%87%E5%AD%97%E8%AF%86%E5%88%AB.html
  上传图片base64编码后进行urlencode要求base64编码和urlencode后大小不超过4M最短边至少15px，最长边最大4096px支持jpg/png/bmp格式
  (Very Important)创建完webapi应用添加合成服务之后一定要设置ip白名单，找到控制台--我的应用--设置ip白名单，如何设置参考：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=41891
  错误码链接：https://www.xfyun.cn/document/error-code (code返回错误码时必看)
  @author iflytek
"""
#-*- coding: utf-8 -*-
import requests
import time
import hashlib
import base64
import json
from asdTools.Secret.key import XFYunAPIKeyOCR, XFYunAPPID
def convert_img_to_text_by_XFYunOCR(img_path:str):
    URL = "http://webapi.xfyun.cn/v1/service/v1/ocr/general"
    APPID = XFYunAPPID
    API_KEY = XFYunAPIKeyOCR
    curTime = str(int(time.time()))
    param = {"language": "cn|en", "location": "false"}
    param = json.dumps(param)
    paramBase64 = base64.b64encode(param.encode('utf-8'))

    m2 = hashlib.md5()
    str1 = API_KEY + curTime + str(paramBase64,'utf-8')
    m2.update(str1.encode('utf-8'))
    checkSum = m2.hexdigest()
    header = {
        'X-CurTime': curTime,
        'X-Param': paramBase64,
        'X-Appid': APPID,
        'X-CheckSum': checkSum,
        'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
    }
    with open(img_path, 'rb') as f:
        f1 = f.read()
    f1_base64 = str(base64.b64encode(f1), 'utf-8')
    data = {'image': f1_base64}
    r = requests.post(URL, data=data, headers=header)
    result = str(r.content, 'utf-8')
    data = json.loads(result)
    text_content = ""
    if 'data' in data and 'block' in data['data']:
        for block in data['data']['block']:
            if 'type' in block and block['type'] == 'text' and 'line' in block:
                for line in block['line']:
                    if 'word' in line:
                        for word in line['word']:
                            if 'content' in word:
                                text_content += word['content']
                                text_content += "\n"
    return text_content
if __name__ == "__main__":
    # 错误码链接：https://www.xfyun.cn/document/error-code (code返回错误码时必看)
    img_path = r"F:\0_DATA\1_DATA\CODE\PYTHON\202304_RJB_C4\ChatGLM\RJB\C4\data\玩游戏时手机发热.png"
    img_path = r"F:\0_DATA\1_DATA\CODE\PYTHON\202304_RJB_C4\ChatGLM\RJB\data\data_7format\people\103.png"
    text = convert_img_to_text_by_XFYunOCR(img_path)
    print(text)
    input("Entry the any key to exit")

