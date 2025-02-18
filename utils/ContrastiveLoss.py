import torch.nn as nn
import torch.nn.functional as F
import torch

class GumbelTopK(nn.Module):

    def __init__(self, k: int, dim: int = -1, gumble: bool = False):

        super().__init__()
        self.k = k
        self.dim = dim
        self.gumble = gumble

    def forward(self, logits):
        if self.gumble:
            u = torch.rand(size=logits.shape, device=logits.device)
            z = - torch.log(- torch.log(u))
            return torch.topk(logits + z, self.k, dim=self.dim)
        else:
            return torch.topk(logits, self.k, dim=self.dim)



class TensorBuffer:

    def __init__(self, buffer_size: int, concat_dim: int, retain_gradient: bool = True):

        self.buffer_size = buffer_size
        self.concat_dim = concat_dim
        self.retain_gradient = retain_gradient
        self.tensor_list = []

    def update(self, tensor):
        if len(self.tensor_list) >= self.buffer_size:
            self.tensor_list.pop(0)
        if self.retain_gradient:
            self.tensor_list.append(tensor)
        else:
            self.tensor_list.append(tensor.detach())

    @property
    def values(self):
        return torch.cat(self.tensor_list, dim=self.concat_dim)


class ContrastiveLoss(nn.Module):

    def __init__(self, temperature: float = 0.1, sample: bool = True, sample_num: int = None):
        super().__init__()
        self.tau = temperature
        self.sample_num = sample_num
        self.sample = False if sample_num is None else sample
        self.topk = GumbelTopK(k=sample_num) if self.sample else None


    def forward(self, v_outputs1_pro, r_outputs1_pro, v_outputs1, r_outputs1):
        B, C, *spatial_size = v_outputs1_pro.shape  # N = H * W * D
        spatial_dims = len(spatial_size)

        #特征图 v_outputs1_pro 和 r_outputs1_pro 进行归一化处理
        norm_v_pro = F.normalize(v_outputs1_pro.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        norm_r_pro = F.normalize(r_outputs1_pro.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        # norm_v_pro = F.normalize(v_outputs1_pro.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(B,-1, C), dim=-1)
        # norm_r_pro = F.normalize(r_outputs1_pro.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(B,-1, C), dim=-1)
        #   图片大小一样的特征 ->

        # sim_matrix = F.cosine_similarity(norm_v_pro.unsqueeze(0), norm_r_pro.unsqueeze(1), dim=2)

        #为每个像素生成一个伪标签，用于确定其前景或背景归属。
        v_max_probs, v_pseudo_lable = torch.softmax(v_outputs1, dim=1).max(dim=1)
        r_max_probs, r_pseudo_lable = torch.softmax(r_outputs1, dim=1).max(dim=1)

        prob_map = (v_max_probs + r_max_probs) / 2
        pred_v = v_pseudo_lable.flatten()  # B * N
        pred_r = r_pseudo_lable.flatten()  # B * N
        prob_map = prob_map.flatten()  # B * N


        v_1_mask = (v_pseudo_lable == 1)
        r_1_mask = (r_pseudo_lable == 1)

        intersection_mask = v_1_mask & r_1_mask
        # intersection_mask = intersection_mask.view(2007040).to(torch.float32)
        intersection_mask = intersection_mask.view(-1).to(torch.float32)
        # intersection_mask = intersection_mask.view(B,-1).to(torch.float32)

        union_mask = v_1_mask | r_1_mask
        # union_mask = union_mask.view(2007040).to(torch.float32)
        union_mask = union_mask.view(-1).to(torch.float32)
        # union_mask = union_mask.view(B,-1).to(torch.float32)

        # sim_matrix = F.cosine_similarity(norm_v_pro, norm_r_pro, dim=-1)
        # nominator1 = torch.exp(intersection_mask * sim_matrix / self.tau)
        # denominator1 = torch.exp((1-intersection_mask) * sim_matrix / self.tau).sum(dim=0) + nominator1
        #
        # negative_sim_matrix = torch.exp(
        #     (1 - intersection_mask) * F.cosine_similarity(norm_v_pro.unsqueeze(0), norm_r_pro.unsqueeze(1),dim=-1) / self.tau)
        # denominator1 = negative_sim_matrix.sum(dim=-1) + nominator1
        #
        # loss1 = -torch.log(nominator1 / (denominator1 + 1e-8)).mean()
        # nominator2 = torch.exp((1-union_mask) * sim_matrix / self.tau)
        # denominator2 = torch.exp(union_mask * sim_matrix / self.tau).sum(dim=0) + nominator2
        # loss2 = -torch.log(nominator2 / (denominator2 + 1e-8)).mean()

        sim_matrix = F.cosine_similarity(norm_v_pro, norm_r_pro, dim=-1) #展平了已经
        neg_sim_matrix = torch.exp(F.cosine_similarity((intersection_mask * norm_v_pro).unsqueeze(0),
                                 (1 - intersection_mask * norm_r_pro).unsqueeze(1), dim=-1) / self.tau)
        nominator1 = torch.exp(intersection_mask * sim_matrix / self.tau)
        denominator1 = neg_sim_matrix.sum(dim=-1)+ nominator1
        loss1 = -torch.log(nominator1 / (denominator1 + 1e-8)).mean()

        nominator2 = torch.exp((1-union_mask) * sim_matrix / self.tau)
        neg_sim_matrix = torch.exp(F.cosine_similarity( (1-union_mask * norm_v_pro).unsqueeze(0),
                                 (union_mask * norm_r_pro).unsqueeze(1), dim=-1) / self.tau)
        denominator2 = neg_sim_matrix.sum(dim=-1) + nominator2
        loss2 = -torch.log(nominator2 / (denominator2 + 1e-8)).mean()

        loss = loss1 + loss2

        return loss
