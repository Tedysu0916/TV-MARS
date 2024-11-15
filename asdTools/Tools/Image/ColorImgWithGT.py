from asdTools.Classes.Image.ImageBase import ImageBase


class ColorImgWithGT(ImageBase):
    """
    使用彩色分割GT为原图上色。示例见 Sample\ColorImgWithGT 
    注：可使用asdTools\Tools\Image\ColorGT.py为GT上色
    Colorize the image with the colorful segmentation GT. Examples can be found in the "Sample\ColorImgWithGT" folder.
    Note: You can use the "asdTools\Tools\Image\ColorGT.py" script to colorize the GT. 
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(multipleFiles=True, **kwargs)

    def __call__(self, imgs_dir:str, GTs_dir:str) -> str:
        self.run(imgs_dir, GTs_dir)
        
    def map_paths_to_fullPaths(self, files_dir:str, files_path:list):
        res = {}
        for file_path in files_path:
            fullPath = file_path
            path = fullPath[len(files_dir)+1:]
            res[path] = fullPath
        return res


    def run(self, imgs_dir:str, GTs_dir:str) -> str:
        # v0.0.14: add fault-tolerant for different suffix.
        imgs_path = self.get_paths_from_dir(imgs_dir)
        imgs_path_originalSuffix = self.filter_ext(imgs_path, ["png", "jpg", "jpeg"])
        imgs_path_mapping_originalSuffix = self.map_paths_to_fullPaths(imgs_dir, imgs_path_originalSuffix)
        GTs_path = self.get_paths_from_dir(GTs_dir)
        GTs_path = self.filter_ext(GTs_path, ["png", "jpg", "jpeg"])
        GTs_path_mapping = self.map_paths_to_fullPaths(GTs_dir, GTs_path)
        for i, (img_path, img_fullPath) in enumerate(imgs_path_mapping_originalSuffix.items()):
            try:
                GT_fullPath = GTs_path_mapping[img_path]
            except:
                GT_fullPath = None
                for other_suffix in ["png", "jpg", "jpeg"]:
                    img_path_otherSuffix = self.convert_fileSuffix(img_path, ["png", "jpg", "jpeg"], other_suffix)
                    if img_path_otherSuffix in GTs_path_mapping.keys():
                        GT_fullPath = GTs_path_mapping[img_path_otherSuffix]
                        break
                if GT_fullPath is None:
                    self.warning(f"{i+1}: GT is not found: {img_path}")
                    continue
                else:
                    self.warning(f'{i+1}: GT: "{img_path}" is not found, but found: "{img_path_otherSuffix}"')
            img_res = self.blend(img_fullPath, GT_fullPath, 0.25)
            img_dir = self.get_dir_of_file(img_path)
            img_name = self.get_name_of_file(img_path, True)
            self.save_image(img_res, output_middle_dir=img_dir, output_file=img_name)
            self.log(f'{i+1}: Img has been colored: "{img_path}"')
        self.done()


if __name__ == "__main__":
    # imgs_dir = r"Sample\ColorImgWithGT\Before\Images"
    # GTs_dir = r"Sample\ColorImgWithGT\Before\Annotations"
    # Converter = ColorImgWithGT()
    # Converter(imgs_dir, GTs_dir)

    imgs_dir = r"F:\0_DATA\1_DATA\Datasets\TTPLA\images\train"
    GTs_dir = r"F:\0_DATA\1_DATA\Datasets\TTPLA\annotations\train"
    Converter = ColorImgWithGT()
    Converter(imgs_dir, GTs_dir)
