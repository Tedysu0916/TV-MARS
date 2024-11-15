from asdTools.Classes.Base.BaseModel import BaseModel
import requests


class APIBase(BaseModel):
    # wait for improvemnt
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def post(self, url:str, data, headers:dict):
        response = requests.post(url, json=data, headers=headers)
        return response

    def parse_response(self, response) -> dict:
        data = response.text
        result = self.convert_val_adaptive(data)
        return result
