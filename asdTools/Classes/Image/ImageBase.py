from asdTools.Classes.Base.BaseModel import BaseModel
from PIL import Image
import numpy as np


class ImageBase(BaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def blend(self, img1:Image.Image, img2:Image.Image, ratio:float):
        # v0.0.14: fix bug from different img mode.
        isTmp_img1 = True if isinstance(img1, str) else False
        isTmp_img2 = True if isinstance(img2, str) else False
        img1 = self.read_img(img1)
        img2 = self.read_img(img2)
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        res = Image.blend(img1, img2, ratio)
        if isTmp_img1:
            img1.close()
        if isTmp_img2:
            img2.close()
        return res

    def convert_img_to_arr(self, img:Image.Image) -> np.ndarray:
        # Image: (Width, Height).
        # Array: (Height, Width, Channel).
        img_array = np.array(img)
        return img_array
    
    def convert_arr_to_img(self, arr:np.ndarray) -> Image.Image:
        img = Image.fromarray(arr.astype(np.uint8))
        return img

    def convert_RGB_to_gray(self, img:Image.Image, need3Channel:bool=False) -> Image.Image:
        gray_img = img.convert('L')
        if need3Channel:
            gray_img = gray_img.convert('RGB')
        return gray_img

    def convert_fileSuffix(self, file_path:str, source_suffixs:list, target_suffix:str):
        for suffix in source_suffixs:
            if file_path.endswith(suffix):
                file_path_new = file_path[:-len(suffix)] + target_suffix
                return file_path_new
        return file_path
        
    def count_img_color(self, img:Image.Image) -> set:
        _img = self.read_img(img)
        unique_color = _img.getcolors(_img.size[0] * _img.size[1])
        _img.close()
        return unique_color

    def count_imgs_color(self, imgs:list) -> dict:
        unique_res = {}
        for img in imgs:
            unique_img = self.count_img_color(img)
            for color in unique_img:
                self.count_in_dict(unique_res, color[1], color[0])
        return unique_res

    def get_img_info(self, img_path:str):
        img = self.read_img(img_path)
        img_array = self.read_img(img, output_type="array")
        img_info = {}
        img_info["path"] = str(img_path)
        img_info["shape"] = img_array.shape
        return img_info
    
    def generate_image(self, mode:str, size:tuple, **kwargs):
        img = Image.new(mode, size, **kwargs)
        return img

    def resize_image(self, 
                     img:Image.Image, 
                     size:tuple=(224, 224)) -> Image.Image:
        img = self.read_img(img)
        img = img.resize(size)
        return img

    def read_img(self, path, output_type:str="Image", h:int=-1, w:int=-1):
        # v0.0.14: add support to resize img.
        if isinstance(path, str):
            with Image.open(path) as f:
                img = f
                img.load()
        elif isinstance(path, Image.Image):
            img = path
        elif isinstance(path, np.ndarray):
            img = self.convert_arr_to_img(path)
        
        if h != -1 and w != -1:
            img = img.resize((w, h), Image.NEAREST)

        if output_type == "Image" or output_type == "image":
            return img
        elif output_type == "Array" or output_type == "array":
            img_array = self.convert_img_to_arr(img)
            img.close()
            return img_array

    def save_image(self, img:Image.Image, output_dir:str="", output_file:str="xxx_resized.png", output_middle_dir:str="", delete=True) -> str:
        img = self.read_img(img)
        output_path = self.generate_output_path(output_dir=output_dir, output_middle_dir=output_middle_dir, output_file=output_file)
        img.save(output_path)
        if delete:
            img.close()
        return output_path



if __name__ == "__main__":
    img_dir = r"F:\0_DATA\1_DATA\Datasets\VITL\seed0\train\gt"
    img_dir = r"F:\0_DATA\1_DATA\Datasets\VITL\seed0\train\vl"
    img_analyzer = ImageBase()
    imgs_path = img_analyzer.get_paths_from_dir(img_dir)
    unique_color = img_analyzer.count_imgs_color(imgs_path)
    pass