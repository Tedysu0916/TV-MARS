from asdTools.Classes.Image.ImageBase import ImageBase
import matplotlib.pyplot as plt
import numpy as np


class VisualizeSegGTDistribution(ImageBase):
    """
    可视化GT分布，GT的类别需要从0开始递增。示例见Sample/VisualizeSegGTDistribution
    Visualize GT distribution, the class ID should be increase progressively from 0. See sample in Sample/VisualizeSegGTDistribution
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(multipleFiles=True, **kwargs)

    def __call__(self, imgs_dir:str="", img_height:int=512, img_width:int=512, device:str="cuda") -> list:
        return self.run(imgs_dir, img_height, img_width, device)

    def run(self, imgs_dir:str="", img_height:int=512, img_width:int=512, device:str="cuda") -> list:
        if imgs_dir == "":
            imgs_dir = self.input("Input folder path:", needLog=True)
        imgs_path = self.get_paths_from_dir(imgs_dir)
        self.log(f"{len(imgs_path)} files in {imgs_dir}.")
        imgs_path = self.filter_ext(imgs_path, ["png", "jpg", "jpeg"])
        self.log(f"{len(imgs_path)} images in {imgs_dir}.")
        color_count_dict = self.count_imgs_color(imgs_path)
        num_classes = len(color_count_dict.keys())
        self.log(f"Number of classes: {num_classes}.")
        self.log(f"Device: {device}")

        if device == "cpu":
            GT_count = np.zeros((img_height, img_width, num_classes))
            for i, img_path in enumerate(imgs_path):
                img = self.read_img(img_path, "Array", h=img_height, w=img_width)
                one_hot = np.eye(num_classes)[img]
                GT_count += one_hot
                self.log(f"{i+1}/{len(imgs_path)}: Get GT distribution for {img_path}")
            GT_count = GT_count.transpose(2, 0, 1)
        else:
            import torch
            GT_count = torch.zeros((num_classes, img_height, img_width), device='cuda')
            for i, img_path in enumerate(imgs_path):
                img = self.read_img(img_path, "Array", h=img_height, w=img_width)
                img_tensor = torch.tensor(img, device='cuda', dtype=torch.int64)
                one_hot = torch.zeros((num_classes, img_height, img_width), device='cuda', dtype=torch.int64)
                one_hot.scatter_(0, img_tensor.unsqueeze(0), 1)
                GT_count += one_hot
                self.log(f"{i+1}/{len(imgs_path)}: Get GT distribution for {img_path}")
            GT_count = GT_count.cpu().numpy()

        GT_maxs = GT_count.max(axis=(1, 2), keepdims=True)
        GT_mins = GT_count.min(axis=(1, 2), keepdims=True)
        GT_diff = GT_maxs - GT_mins
        GT_diff[GT_diff == 0] = 1
        GT_distribution = (GT_count - GT_mins) / GT_diff
        GT_distribution = GT_distribution * 1.5 - 0.25
        
        GT_infos = {}

        for i in range(num_classes):
            GT_infos[f"{i}"] = {
                "max count in single pixel": GT_maxs[i, 0, 0],
                "min count in single pixel": GT_mins[i, 0, 0],
                "sum of pixel": GT_count[i, :, :].sum(),
            }
            GT_show = GT_distribution[i]
            GT_show = plt.get_cmap('rainbow')(GT_show)[..., :3] 
            GT_show = (GT_show * 255).astype(np.uint8)
            save_path_GTDistribution = self.save_image(GT_show, output_file=f"class-{i}.png")
            self.log(f"{i+1}/{num_classes}: Save GT distribution of class-{i} to {save_path_GTDistribution}")
        save_path_GTInfo = self.generate_output_path(output_file="GT_infos.json")
        self.save_file(GT_infos, log_path=save_path_GTInfo)
        self.log(f"GT infos saved to {save_path_GTInfo}")
        self.done()
        

if __name__ == "__main__":
    # Slow, use cpu.
    # imgs_dir = r"Sample\VisualizeGTDistribution\Sample1-before+after\before"
    # VisualizeSegGTDistribution()(imgs_dir, img_height=512, img_width=512, device="cpu")

    # Fast, use gpu (cuda) by torch.
    imgs_dir = r"Sample\VisualizeGTDistribution\Sample1-before+after\before"
    VisualizeSegGTDistribution()(imgs_dir, img_height=512, img_width=512)
    
    # imgs_dir = r"F:\0_DATA\1_DATA\Datasets\ADE20K\ADEChallengeData2016\annotations\training"
    # VisualizeGTDistribution()(imgs_dir)