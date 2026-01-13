from torch.nn import init
from torchvision import models

from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn
from collections import OrderedDict

class TMarNet(nn.Module):
    def __init__(self, args, num_classes=625):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        # print('----------------', self.embed_dim)

        self.logit_scale = torch.ones([]) * (1 / args.temperature) #生成一个标量张量

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'msc' in args.loss_names:
            # self.embed_dim = args.embed_dim
            heads = self.embed_dim // 64

            # 1. 局部对齐模块
            self.local_attn = nn.MultiheadAttention(
                self.embed_dim,
                heads,
                batch_first=True
            )
            self.local_norm_t = LayerNorm(self.embed_dim)
            self.local_norm_v = LayerNorm(self.embed_dim)

            # 2. 全局对齐模块(保持原有)
            self.cross_attn = nn.MultiheadAttention(
                self.embed_dim,
                heads,
                batch_first=True
            )
            self.cross_modal_transformer = Transformer(
                width=self.embed_dim,
                layers=args.cmt_depth,
                heads=heads
            )

            # 3. Layer Norms
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            # 4. 特征融合层
            self.fusion = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                QuickGELU(),
                LayerNorm(self.embed_dim)
            )
            self.alpha = nn.Parameter(torch.FloatTensor([0.5]).half())

            # 5. MLM头
            self.mlm_head = nn.Sequential(
                OrderedDict([
                    ('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                    ('gelu', QuickGELU()),
                    ('ln', LayerNorm(self.embed_dim)),
                    ('fc', nn.Linear(self.embed_dim, args.vocab_size))
                ])
            )

            # 6. 初始化权重
            scale = self.cross_modal_transformer.width ** -0.5
            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5

            # 初始化transformer
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

                # 初始化cross attention
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            # 初始化local attention
            nn.init.normal_(self.local_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.local_attn.out_proj.weight, std=proj_std)

            # 初始化fusion层
            nn.init.normal_(self.fusion[0].weight, std=fc_std)

            # 初始化MLM头
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')


    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:,0,:].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption']
        b,t,c,h,w = images.shape
        images = images.view(-1,c,h,w)
        image_feats, text_feats = self.base_model(images, caption_ids)#[BT,129,512] [B,77,512]

        # i_feats = image_feats.view(b, t, -1).mean(1).float()  # [16,512] For resnet
        i_feats = image_feats[:,0,:].view(b, t, -1).mean(1).float()  # [16,512]
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'vtc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})

        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})
        # if 'TSDA' in self.current_task:
        #     ret.update({'TSDA_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})


        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()

            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()

            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'msc' in self.current_task:
        
            mlm_ids = batch['mlm_ids']
            mlm_feats = self.base_model.encode_text(mlm_ids)
            # b, t = batch['image_feats'].shape[:2]


            local_img_feats = image_feats.view(b, t, -1, self.embed_dim)  # [B, T, N, D]
            text_local = self.local_norm_t(mlm_feats)

            local_feats = []
            
            for i in range(t):
                img_local = self.local_norm_v(local_img_feats[:, i])  # [B, N, D]
                local_out, _ = self.local_attn(
                    text_local, img_local, img_local,
                    need_weights=False
                )
                local_feats.append(local_out)

            local_feats = torch.stack(local_feats, dim=1)  # [B, T, L, D]
            local_feats = local_feats.mean(1)  # [B, L, D]


            mlm_img_feats = image_feats.view(b, t, -1, self.embed_dim).mean(1)
            global_feats = self.cross_former(mlm_feats, mlm_img_feats, mlm_img_feats)

            alpha = torch.sigmoid(self.alpha)  # 确保在0-1之间
            fused_feats = self.fusion(
                torch.cat([
                    local_feats * alpha,
                    global_feats * (1 - alpha)
                ], dim=-1)
            )


            x = self.mlm_head(fused_feats)
            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)

            ret.update({
                'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight
            })

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret


def build_model(args, num_classes=11003):
    model = TMarNet(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
