
import torch
from math import pi

class RotatE:
    def __init__(self, k, max_rel_size=None, entity_embedding=None, relation_embedding=None):
        self.internal_k = 2 * k
        self.max_rel_size = max_rel_size
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding

    def __call__(self, e_s_id, e_p_id):
        e_s = self.entity_embedding[e_s_id]
        e_p = self.relation_embedding[e_p_id]
        e_s_real, e_s_img = torch.chunk(e_s, 2, axis=0)
        theta_pred, _ = torch.chunk(e_p, 2, axis=0)

        embedding_range = (6 / (self.internal_k * self.max_rel_size)) ** 0.5
        e_p_real = torch.cos(theta_pred / (embedding_range / pi))
        e_p_img = torch.sin(theta_pred / (embedding_range / pi))

        e_o_real = e_s_real * e_p_real - e_s_img * e_p_img
        e_o_img = e_s_real * e_p_img + e_s_img * e_p_real
        return torch.cat([e_o_real, e_o_img], axis=0)