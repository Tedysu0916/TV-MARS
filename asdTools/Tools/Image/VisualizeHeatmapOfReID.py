from torchvision.transforms.functional import to_pil_image

import sys
import os
# 获取当前脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
# 将项目根目录添加到sys.path中
sys.path.append(project_root)


from asdTools.Classes.Image.ImageBase import ImageBase
from torchcam.utils import overlay_mask
from torchcam.methods import GradCAM

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer

class VisualizeHeatmapOfReID(ImageBase):
    """ Sample: Sample/VisualizeHeatmapOfReID
    使用torch-cam可视化热力图，仅测试于ReID模型。
    Visualize heatmap by torch-cam, test only on ReID.
    
    torch-cam: `pip install torchcam` or `conda install -c frgfm torchcam`, GitHub: https://github.com/frgfm/torch-cam
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(multipleFiles=True, **kwargs)

    def __call__(self, 
            model, 
            imgs_dir:str, 
            transform,
            img_ext:list=["png", "jpg", "jpeg"],
            torchCamMethod=GradCAM,
            device="cuda") -> str:
        self.run(model, imgs_dir, transform, img_ext, torchCamMethod, device)
        
    def run(self, model, imgs_dir:str, transform, img_ext:list, torchCamMethod, device) -> str:
        self.begining()
        # get paths of imgs from img_dir
        imgs_path = self.get_paths_from_dir(imgs_dir)
        self.log(f"{len(imgs_path)} files found in {imgs_dir}")
        imgs_path = self.filter_ext(imgs_path, img_ext)
        self.log(f"{len(imgs_path)} images found after filter extension by {img_ext}")
        # init torchCam
        model.to(device).eval()

        # todo edit
        # 使用模型中的一个卷积层作为目标层
        target_layer_name = 'layer4.0.bn1'#"layer3.5.conv3" NL_3.5.kan_cube  layer2.0 conv1
        # target_layer_name = 'layer4.1.conv3'  # "layer3.5.conv3"

        target_layer = get_layer_from_name(model.backbone, target_layer_name)

        # cam_extractor = torchCamMethod(model,input_shape=(3, 384, 128),target_layer = target_layer)
        cam_extractor = torchCamMethod(model, input_shape=(3, 256, 256), target_layer=target_layer)

        for i, img_path in enumerate(imgs_path):
            # model(x)
            img = self.read_img(img_path)
            x = transform(img).unsqueeze(0).to(device)
            print("***************** shape = ",x.shape)
            out = model(x)
            # visualize feature
            activation_map = cam_extractor(class_idx=0, scores=out.unsqueeze(0))[0]
            result = overlay_mask(img, to_pil_image(activation_map, mode='F'), alpha=0.5)
            # save img
            save_path = self.remove_root_of_path(path=img_path, root=imgs_dir)
            save_middle_dir = self.get_dir_of_file(save_path)
            save_name = self.get_name_of_file(save_path, True)
            save_path = self.save_image(result, output_middle_dir=save_middle_dir, output_file=save_name)
            self.log(f"{i+1}/{len(imgs_path)}: the heatmap of {img_path} has been saved to {save_path}.")
        self.done()

def get_layer_from_name(model, layer_name):
    layers = layer_name.split('.')
    current_layer = model
    for layer in layers:
        if hasattr(current_layer, layer):
            current_layer = getattr(current_layer, layer)
        else:
            raise ValueError(f"Layer {layer} not found in the model backbone.")
    return current_layer

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)



    return cfg

if __name__ == "__main__":
    from asdTools.Tools.Image.VisualizeHeatmapOfReID import VisualizeHeatmapOfReID
    from torchvision import transforms as T
    # import data.img_transforms as T
    import torch
    import torch.nn.functional as F
    
    imgs_dir = "/media/jqzhu/e/simin/fastreid/CCBR/datasets/veri/image_test"  #simiiii
    # imgs_dir = "/media/jqzhu/e/simin/fastreid/CCBR/datasets/MSMT17_V1/test"  # simiiii
    # imgs_dir = "/media/jqzhu/e/simin/fastreid/CCBR/datasets/Market-1501-v15.09.15/bounding_box_test"

    # weight = "logs/0/baseline.pth.tar"
    # checkpoint = torch.load(weight)
    # model = ResNet50(config)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.cuda().eval()

    args = default_argument_parser().parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = True
    model = DefaultTrainer.build_model(cfg)
    # print("---------------", cfg.MODEL.WEIGHTS)
    cfg.MODEL.WEIGHTS = '/media/jqzhu/e/simin/fastreid/CCBR/ccbr/VeRi776/model_final.pth'
    # cfg.MODEL.WEIGHTS = '/media/jqzhu/e/simin/fastreid/CCBR/ccbr/msmt17/model_final.pth'
    # cfg.MODEL.WEIGHTS = '/media/jqzhu/e/simin/fastreid/CCBR/ccbr/market1501/model_final.pth'
    #  ------> todo edit

    # state_dict = torch.load(cfg.MODEL.WEIGHTS)
    # model.load_state_dict(state_dict)
    # print(state_dict.keys())

    # model = Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
    # checkpointer = Checkpointer(model)
    # checkpoint = checkpointer.load(cfg.MODEL.WEIGHTS)
    # print('123123',checkpoint['m'])

    # 提取模型权重
    # state_dict = checkpoint['model']  # 可能需要检查字典键值，确保是 'model'
    state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location='cuda')
    # print("************************************ Loaded State Dict Keys:", state_dict.keys())
    # model.load_state_dict(state_dict,strict=False)
    model.load_state_dict(state_dict['model'])
    # model_state_dict = model.state_dict()

    # res = DefaultTrainer.test(cfg, model)
    # model = res

    
    transform_test = T.Compose([
        # T.Resize((384, 128)),
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    color_heatmap = VisualizeHeatmapOfReID()
    color_heatmap(model, imgs_dir, transform_test)

