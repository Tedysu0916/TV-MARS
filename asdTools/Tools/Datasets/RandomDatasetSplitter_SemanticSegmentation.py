from asdTools.Classes.Image.ImageBase import ImageBase


class RandomDatasetSplitter_SemanticSegmentation(ImageBase):
    """
    对内部文件名相同的两个文件夹进行随机划分，如语义分割中的RGB图与GT图。
    Randomly split two folders with the same internal file names, such as RGB images and GT (Ground Truth) images in semantic segmentation.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(multipleFiles=True, **kwargs)

    def __call__(self, 
                 dir1:str, 
                 dir2:str, 
                 ratios:tuple,
                 output_dir:str=""):
        return self.run(dir1, dir2, ratios, output_dir)

    def run(self, 
            dir1:str, 
            dir2:str, 
            ratios:tuple,
            output_dir:str=""):
        self.log("Start to split datasets.")
        self.log(f"Dir1: {dir1}")
        self.log(f"Dir2: {dir2}")
        files_path1 = self.get_paths_from_dir(dir1)
        files_path2 = self.get_paths_from_dir(dir2)
        self.log(f"{len(files_path1)} files in dir1, {len(files_path2)} files in dir2.")
        ext_list = ["png", 'jpg', 'jpeg']
        files_path1 = self.filter_ext(files_path1, ext_list=ext_list)
        files_path2 = self.filter_ext(files_path2, ext_list=ext_list)
        self.log(f"After ext fliter: {ext_list}, {len(files_path1)} files in dir1, {len(files_path2)} files in dir2.")
        if len(files_path1) != len(files_path1):
            self.raise_error("The number of two dir are not correct, exit.")
        files_name1 = self.get_name_of_files(files_path1, False)
        files_name2 = self.get_name_of_files(files_path2, False)
        if not self.check_equal(files_name1, files_name2):
            name1_name2 = self.merge_2list_to_dcit(files_name1, files_name2)
            match_path = self.generate_output_path(output_dir, output_file="name_match.json")
            match_path = self.save_file(name1_name2, match_path)
            self.raise_error(f"Files name are not correct, the match file saved in {match_path}, exit.")
        
        lists_split = self.split_list(files_name1, ratios)
        name_path1 = self.merge_2list_to_dcit(files_name1, files_path1)
        name_path2 = self.merge_2list_to_dcit(files_name2, files_path2)
        for i, listt in enumerate(lists_split):
            self.log(f"----- Start to copy files, ratio: {ratios[i]}, length: {len(listt)} -----")
            for j, name in enumerate(listt):
                file1_src = name_path1[name]
                file2_src = name_path2[name]
                file1_name = self.get_name_of_file(file1_src, True)
                file2_name = self.get_name_of_file(file2_src, True)
                file1_dest = self.generate_output_path(output_middle_dir=f"{i}_{ratios[i]}_dir1", output_file=file1_name)
                file2_dest = self.generate_output_path(output_middle_dir=f"{i}_{ratios[i]}_dir2", output_file=file2_name)
                self.copy(file1_src, file1_dest)
                self.copy(file2_src, file2_dest)
                self.log(f"Split {i} with {ratios[i]} ratio: {j}: {file1_src} is copied to {file1_dest}, {file2_src} is copied to {file2_dest}")
            self.log(f"----- End of coping files, ratio: {ratios[i]}, length: {len(listt)} -----")
        message_done = [f"Dataset is splitted to {len(lists_split)} set"]
        for i in range(len(lists_split)):
            message_done.append(f"Set{i}: ratio={ratios[i]}, length={len(lists_split[i])}" )
        self.done(message_done)

if __name__ == "__main__":
    dir1 = r"F:\0_DATA\1_DATA\Datasets\VITLD\ori_vl"
    dir2 = r"F:\0_DATA\1_DATA\Datasets\VITLD\ori_gt"
    ratios = (0.7, 0.2, 0.1)
    RandomDatasetSplitter_SemanticSegmentation()(dir1, dir2, ratios)