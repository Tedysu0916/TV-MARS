import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_video_sdm(video_features, text_features, pid, logit_scale, video_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching for Video Sequences
    Args:
    video_features: shape [batch_size, num_frames, feature_dim] # 例如 [4, 8, 512]
    text_features: shape [batch_size, feature_dim] # 例如 [4, 512]
    pid: person ID, shape [batch_size]
    logit_scale: scaling factor for logits
    video_id: optional video ID for soft label creation
    factor: factor for mixing PID and video ID labels
    epsilon: small value to prevent log(0)
    """
    batch_size, num_frames, feature_dim = video_features.shape
    # 重塑video_features以处理帧序列
    video_features = video_features.view(-1, feature_dim)  # [batch_size * num_frames, feature_dim]

    # 扩展text_features以匹配视频帧数量
    text_features = text_features.unsqueeze(1).repeat(1, num_frames, 1)  # [batch_size, num_frames, feature_dim]
    text_features = text_features.view(-1, feature_dim)  # [batch_size * num_frames, feature_dim]

    # 扩展pid以匹配帧数量
    expanded_pid = pid.repeat_interleave(num_frames).reshape(-1, 1)  # [batch_size * num_frames, 1]
    pid_dist = expanded_pid - expanded_pid.t()
    labels = (pid_dist == 0).float()

    if video_id is not None:
        expanded_video_id = video_id.repeat_interleave(num_frames).reshape(-1, 1)
        video_id_dist = expanded_video_id - expanded_video_id.t()
        video_id_mask = (video_id_dist == 0).float()
        labels = (labels - video_id_mask) * factor + video_id_mask

        # 特征归一化
    video_norm = video_features / video_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)

    # 计算相似度
    t2v_cosine_theta = text_norm @ video_norm.t()
    v2t_cosine_theta = t2v_cosine_theta.t()

    # 应用logit scale
    text_proj_video = logit_scale * t2v_cosine_theta
    video_proj_text = logit_scale * v2t_cosine_theta

    # 归一化真实匹配分布
    labels_distribute = labels / labels.sum(dim=1)

    # 计算损失
    v2t_pred = F.softmax(video_proj_text, dim=1)
    v2t_loss = v2t_pred * (F.log_softmax(video_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2v_pred = F.softmax(text_proj_video, dim=1)
    t2v_loss = t2v_pred * (F.log_softmax(text_proj_video, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(v2t_loss, dim=1)) + torch.mean(torch.sum(t2v_loss, dim=1))

    return loss
def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """

    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()

    labels = (pid_dist == 0).float()


    if image_id != None:
        print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


