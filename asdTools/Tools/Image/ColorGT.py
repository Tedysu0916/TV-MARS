from asdTools.Classes.Image.ImageBase import ImageBase


class ColorGT(ImageBase):
    """
    为GT图片上色，支持RGB图与灰度图的输入及输出。示例见Sample/ColorGT。
    Colorize GT images. Support input and output RGB images and grayscale images.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(multipleFiles=True, **kwargs)
        self.ade20k_color = ((0, 0, 0),
                             (120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50),
                             (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255),
                             (230, 230, 230), (4, 250, 7), (224, 5, 255), (235, 255, 7),
                             (150, 5, 61), (120, 120, 70), (8, 255, 51), (255, 6, 82),
                             (143, 255, 140), (204, 255, 4), (255, 51, 7), (204, 70, 3),
                             (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255),
                             (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220),
                             (255, 9, 92), (112, 9, 255), (8, 255, 214), (7, 255, 224),
                             (255, 184, 6), (10, 255, 71), (255, 41, 10), (7, 255, 255),
                             (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
                             (255, 122, 8), (0, 255, 20), (255, 8, 41), (255, 5, 153),
                             (6, 51, 255), (235, 12, 255), (160, 150, 20), (0, 163, 255),
                             (140, 140, 140), (250, 10, 15), (20, 255, 0), (31, 255, 0),
                             (255, 31, 0), (255, 224, 0), (153, 255, 0), (0, 0, 255),
                             (255, 71, 0), (0, 235, 255), (0, 173, 255), (31, 0, 255),
                             (11, 200, 200), (255, 82, 0), (0, 255, 245), (0, 61, 255),
                             (0, 255, 112), (0, 255, 133), (255, 0, 0), (255, 163, 0),
                             (255, 102, 0), (194, 255, 0), (0, 143, 255), (51, 255, 0),
                             (0, 82, 255), (0, 255, 41), (0, 255, 173), (10, 0, 255),
                             (173, 255, 0), (0, 255, 153), (255, 92, 0), (255, 0, 255),
                             (255, 0, 245), (255, 0, 102), (255, 173, 0), (255, 0, 20),
                             (255, 184, 184), (0, 31, 255), (0, 255, 61), (0, 71, 255),
                             (255, 0, 204), (0, 255, 194), (0, 255, 82), (0, 10, 255),
                             (0, 112, 255), (51, 0, 255), (0, 194, 255), (0, 122, 255),
                             (0, 255, 163), (255, 153, 0), (0, 255, 10), (255, 112, 0),
                             (143, 255, 0), (82, 0, 255), (163, 255, 0), (255, 235, 0),
                             (8, 184, 170), (133, 0, 255), (0, 255, 92), (184, 0, 255),
                             (255, 0, 31), (0, 184, 255), (0, 214, 255), (255, 0, 112),
                             (92, 255, 0), (0, 224, 255), (112, 224, 255), (70, 184, 160),
                             (163, 0, 255), (153, 0, 255), (71, 255, 0), (255, 0, 163),
                             (255, 204, 0), (255, 0, 143), (0, 255, 235), (133, 255, 0),
                             (255, 0, 235), (245, 0, 255), (255, 0, 122), (255, 245, 0),
                             (10, 190, 212), (214, 255, 0), (0, 204, 255), (20, 0, 255),
                             (255, 255, 0), (0, 153, 255), (0, 41, 255), (0, 255, 204),
                             (41, 0, 255), (41, 255, 0), (173, 0, 255), (0, 245, 255),
                             (71, 0, 255), (122, 0, 255), (0, 255, 184), (0, 92, 255),
                             (184, 255, 0), (0, 133, 255), (255, 214, 0), (25, 194, 194),
                             (102, 255, 0), (92, 0, 255))

    def __call__(self, imgs_dir:str="", mapping_path:str="") -> list:
        return self.run(imgs_dir, mapping_path)

    def run(self, imgs_dir:str="", mapping_path:str="") -> list:
        # v0.0.14: fix bug from shape: np.array (H, W, C) and Image.Image (W, H)
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

        self.log('If you need to specify the conversion results for each color, ')
        self.log('please create a JSON file in the format of {"before": "after"}, ')
        self.log('For example {"0": "0", "255": "1"} for grayscale images, ')
        self.log('and {"(0, 0, 0)": "0" , "(255, 255, 255)": "1"} for RGB images.')
        self.log("If you don't need to specify the color mapping, please leave it blank.")
        if mapping_path == "":
            if len(self.ade20k_color) < len(color_count_dict):
                self.done()
                self.raise_error(f"There are too many color types, and the automatic coloring uses {len(self.ade20k_color)} colors from ADE20K. The current number of colors is {len(color_count_dict)}. Please increase the self.ade20k_color colors or use a color mapping JSON.")
            color_mapping = {}
            for i, k in enumerate(color_count_dict.keys()):
                color_mapping[k] = self.ade20k_color[i]
        else:
            color_mapping = self.read_json(mapping_path)
            color_mapping = self.convert_val_adaptive(color_mapping)

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
                img_res = self.generate_image("RGB", (img_array.shape[1], img_array.shape[0]))
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
    # mapping_path = r"Sample\ColorGT\Sample1_withMapping\before\color_mapping.json"
    # imgs_dir = r"Sample\ColorGT\Sample1_withMapping\before"
    # ColorGT()(imgs_dir, mapping_path=mapping_path)

    # imgs_dir = r"Sample\ColorGT\Sample2_withoutMapping\before"
    # ColorGT()(imgs_dir)
    
    imgs_dir = r"F:\0_DATA\1_DATA\Datasets\TTPLA\annotations\train"
    ColorGT()(imgs_dir)