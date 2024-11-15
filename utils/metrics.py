import shutil

from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True,captions=None,image_path=None):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    # # --------------------------Visualization on T2V----------------------------------------
    # indices_idxs = indices

    # if captions is not None and image_path is not None:
    #     for q_id in range(len(indices_idxs)):
    #         # 为每个查询创建一个目录
    #         query_dir = os.path.join('TV-MARS-visualization', f'q{q_id}')
    #         os.makedirs(query_dir, exist_ok=True)
    #
    #         # 保存查询文本到caption.txt
    #         with open(os.path.join(query_dir, 'caption.txt'), 'w', encoding='utf-8') as f:
    #             f.write(captions[q_id])
    #
    #             # 处理前20个检索结果
    #         for g_id in range(min(20, len(indices_idxs[q_id]))):
    #             q_pid = q_pids[q_id]
    #             g_pid = g_pids[indices_idxs[q_id][g_id]]
    #
    #             # 确定是否匹配
    #             match_status = "match" if q_pid == g_pid else "mismatch"
    #
    #             # 创建检索结果目录
    #             result_dir = os.path.join(query_dir, f'g{g_id}_{match_status}')
    #             os.makedirs(result_dir, exist_ok=True)
    #
    #             # 获取图像路径列表
    #             g_img_paths = image_path[indices_idxs[q_id][g_id]]
    #
    #             # 如果是单个字符串路径，转换为列表
    #             if isinstance(g_img_paths, str):
    #                 g_img_paths = [g_img_paths]
    #
    #                 # 复制所有相关图片到目标目录
    #             for idx, img_path in enumerate(g_img_paths):
    #                 if not os.path.exists(img_path):
    #                     continue
    #
    #                     # 构建目标文件名
    #                 filename = f'frame_{idx:03d}.jpg'
    #                 dst_path = os.path.join(result_dir, filename)
    #
    #                 # 复制图片
    #                 shutil.copy(img_path, dst_path)

    return all_cmc, mAP, mINP, indices



class Evaluator():
    def __init__(self, val_loader):
        self.val_loader = val_loader
        self.logger = logging.getLogger("Tmars.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats,captions,image_path = [], [], [], [], [], []
        # text
        for batch in self.val_loader:
            pid = batch['pids'].to(device)
            img = batch['images'].to(device)
            caption = batch['caption'].to(device)

            
            if len(img.shape)==6: # todo: dense mode!!!!
                _,_,t,c,h,w = img.shape
                _,_,d1 = caption.shape
                _,_,d2 = pid.shape

                img = img.view(-1,t,c,h,w)
                caption = caption.view(-1,d1)
                pid = pid.view(-1,d2)

            b, t, c, h, w = img.shape
            img = img.view(-1, c, h, w)


            with torch.no_grad():
                img_feat = model.encode_image(img).view(b,t,-1).mean(1)
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten
            gids.append(pid.view(-1))  # flatten
            gfeats.append(img_feat)
            qfeats.append(text_feat)
            #visualization choose
            # for sublist in batch['cname']:
            #     captions.append(sublist[0])
            # image_path += batch['img_paths']
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids,captions,image_path

    def eval(self, model, i2t_metric=False):

        # qfeats, gfeats, qids, gids,captions,image_path = self._compute_embedding(model)#visualzation choose
        qfeats, gfeats, qids, gids,_,_ = self._compute_embedding(model)

        print('-------check info:',qfeats.shape,gfeats.shape,qids.shape,gids.shape)

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()

        # t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True,captions=captions,image_path=image_path)#visualzation choose
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        # t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        # 先将张量移到CPU，然后再转换为numpy
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.cpu().numpy(), t2i_mAP.cpu().numpy(), t2i_mINP.cpu().numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])

        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))

        return t2i_cmc[0]
