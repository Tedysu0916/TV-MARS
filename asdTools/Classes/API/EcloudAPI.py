from asdTools.Secret.key import CMSSEcloudOcrClient_access_key, CMSSEcloudOcrClient_secret_key
from asdTools.Classes.API.APIBase import APIBase
from ecloud import CMSSEcloudOcrClient


class EcloudAPI(APIBase):
    # wait for improvemnt
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.access_key = CMSSEcloudOcrClient_access_key
        self.secret_key = CMSSEcloudOcrClient_secret_key
        self.url = 'https://api-wuxi-1.cmecloud.cn:8443'

    def request_webimage(self, img_path:str):
        # ref: https://ecloud.10086.cn/op-help-center/doc/article/40774
        requesturl = '/api/ocr/v1/webimage'
        try:
            ocr_client = CMSSEcloudOcrClient(self.access_key, self.secret_key, self.url)
            response = ocr_client.request_ocr_service_file(requestpath=requesturl, imagepath=img_path)
            return response
        except ValueError as e:
            print(e)

    def convert_img_to_str_by_OCR(self, img_path:str):
        response = self.request_webimage(img_path)
        data = self.parse_response(response)
        results_OCR = data["body"]["content"]["prism_wordsInfo"]
        result = ""
        for res in results_OCR:
            word = res["word"]
            result += word
            result += "\n"
        return result

if __name__ == "__main__":
    api = EcloudAPI()
    img = r"F:\0_DATA\1_DATA\CODE\PYTHON\202304_RJB_C4\ChatGLM\ChatGLM\data\1.png"
    message = api.request_webimage(img)
    api.log(message)