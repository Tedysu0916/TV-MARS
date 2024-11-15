from asdTools.Classes.Image.ImageBase import ImageBase
import sys


class ImageResizer_256x144(ImageBase):
    """
    调整图片的分辨率至256x144，提供了Windows右键菜单快捷方式。
    Resize the resolution of the image to 256x144 and provides a Windows right-click menu shortcut.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(multipleFiles=True, **kwargs)

    def __call__(self, 
                 img_path:str, 
                 size:tuple=(256, 144), 
                 output_dir:str="", 
                 output_file:str="") -> str:
        img_resize_path = self.run(img_path, size, output_dir, output_file)
        return img_resize_path

    def run(self, 
            img_path:str, 
            size:tuple=(256, 144), 
            output_dir:str="", 
            output_file:str="") -> str:
        img = self.read_img(img_path)
        img_resize = self.resize_image(img, size)
        if output_dir == "":
            output_dir = self.get_dir_of_file(img_path)
            output_dir = self.join(output_dir, f"{size[0]}x{size[1]}")
        if output_file == "":
            output_file = self.get_name_of_file(img_path, True)
            output_file = self.add_suffix(output_file, "_resized")
        img_resize_path = self.save_image(img_resize, output_dir, output_file)
        img.close()
        img_resize.close()
        return img_resize_path



if __name__ == "__main__":
    try:
        img_path = sys.argv[1]
        img_path = img_path.strip('"')
        size = (256, 144)
        Resizer = ImageResizer_256x144(name=f"ImageResizer_{size[0]}x{size[1]}")

        output_path = Resizer(img_path, size)
        # if you need log, use the following code.
        # Resizer.log(f"Resized {img_path} to {size[0]}x{size[1]}, saved to {output_path}.")
    except Exception as e:
        print(e)
        input("Press Enter to exit.")