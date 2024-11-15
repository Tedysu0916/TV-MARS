from asdTools.Classes.Image.ImageBase import ImageBase


class ConvertGT2MMSeg(ImageBase):
    """
    将GT图片转为MMSegmentation所需的图片格式。支持RGB图与灰度图的输入。示例见Sample\ConvertGT2MMSeg
    Convert GT images to the image format required by MMSegmentation. Support input RGB images and grayscale images. EXample in Sample\ConvertGT2MMSeg
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(multipleFiles=True, **kwargs)

    def __call__(self, imgs_dir:str="") -> list:
        return self.run(imgs_dir)

    def run(self, imgs_dir:str="") -> list:
        if imgs_dir == "":
            imgs_dir = self.input("Input folder path:", needLog=True)
        imgs_path = self.get_paths_from_dir(imgs_dir)
        self.log(f"{len(imgs_path)} files in {imgs_dir}.")
        imgs_path = self.filter_ext(imgs_path, ["png", "jpg", "jpeg"])
        self.log(f"{len(imgs_path)} images in {imgs_dir}.")

        self.log("Start to analyze images color...")
        color_count_dict = self.count_imgs_color(imgs_path)
        self.log(f"These images has {len(color_count_dict)} pixel value:")
        self.log(color_count_dict)
        color_count_path = self.generate_output_path(output_file=f"color_count.json")
        color_count_str = self.convert_json_to_str(color_count_dict)
        self.save_file(color_count_str, color_count_path)

        color_mapping = {}
        for i, k in enumerate(color_count_dict.keys()):
            color_mapping[k] = i

        color_mapping_path = self.generate_output_path(output_file=f"color_mapping.json")
        color_mapping_str = self.convert_json_to_str(color_mapping)
        value_first = next(iter(color_mapping.values()))
        if isinstance(value_first, tuple) and len(value_first) == 3:
            generateRGB = True
        else:
            generateRGB = False
        self.save_file(color_mapping_str, color_mapping_path)
        self.log(f"Color mapping is saved in {color_mapping_path}.")
        imgs_res_path = {}
        for k, img_path in enumerate(imgs_path):
            img_array = self.read_img(img_path, "array")
            if generateRGB:
                img_res = self.generate_image("RGB", (img_array.shape[0], img_array.shape[1]))
                img_res = self.read_img(img_res, "array")
            else:
                img_res = self.generate_image("L", (img_array.shape[0], img_array.shape[1]))
            for i in range(img_array.shape[0]):
                for j in range(img_array.shape[1]):
                    try:
                        val_ori = tuple(img_array[i][j])
                    except:
                        val_ori = int(img_array[i][j])
                    val_new = color_mapping[val_ori]
                    if generateRGB:
                        img_res[i][j][0] = val_new[0]
                        img_res[i][j][1] = val_new[1]
                        img_res[i][j][2] = val_new[2]
                    else:
                        img_res.putpixel((j, i), val_new)
            img_res_name = self.get_name_of_file(img_path, True)
            img_res_dir = self.get_dir_of_file(img_path, root=imgs_dir)
            img_res_path = self.save_image(img_res, output_file=img_res_name, output_middle_dir=img_res_dir)
            del img_res
            del img_array
            imgs_res_path[img_path] = img_res_path
            self.log(f"{k+1}: The MMSeg format of {img_path} is saved in {img_res_path}")
        self.done()
        return imgs_res_path
        

if __name__ == "__main__":
    imgs_dir = r"Sample\ConvertGT2MMSeg\before"
    ConvertGT2MMSeg()(imgs_dir)