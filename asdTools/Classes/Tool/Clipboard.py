import pyperclip


class Clipboard():
    def copy(data:str):
        pyperclip.copy(data)