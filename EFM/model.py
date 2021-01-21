import torch
from torch import log, neg_
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

class _ATTR_NETWORK(nn.Module):
    def __init__(self, vocab_obj, args, device):
        super(_ATTR_NETWORK, self).__init__()

        self.m_device = device
        
        self.m_vocab_size = vocab_obj.vocab_size
        self.m_user_num = vocab_obj.user_num
        self.m_item_num = vocab_obj.item_num

        self.m_tag_embed_size = args.attr_emb_size
        self.m_user_embed_size = args.user_emb_size
        self.m_item_embed_size = args.item_emb_size

        #self.m_tag_user_embedding = nn.Embedding(self.m_vocab_size, self.m_tag_embed_size)
        #self.m_tag_item_embedding = nn.Embedding(self.m_vocab_size, self.m_tag_embed_size)
        self.m_tag_user_embedding = nn.Linear(self.m_tag_embed_size, self.m_vocab_size)
        self.m_tag_item_embedding = nn.Linear(self.m_tag_embed_size, self.m_vocab_size)
        self.dropout = nn.Dropout(p=0.1)

        self.m_user_embedding = nn.Embedding(self.m_user_num, self.m_user_embed_size)
        self.m_item_embedding = nn.Embedding(self.m_item_num, self.m_item_embed_size)

        self.m_H1 = nn.Embedding(self.m_user_num, 32)
        self.m_H2 = nn.Embedding(self.m_item_num, 32)

        self.f_init_weight()

        self = self.to(self.m_device)

    def f_init_weight(self):
        torch.nn.init.normal_(self.m_tag_user_embedding.weight, 0.0, 0.01)
        torch.nn.init.normal_(self.m_tag_item_embedding.weight, 0.0, 0.01)
        torch.nn.init.normal_(self.m_user_embedding.weight, 0.0, 0.01)
        torch.nn.init.normal_(self.m_item_embedding.weight, 0.0, 0.01)


    def forward(self, pos_tag_input, neg_tag_input, rating, user_ids, item_ids):

        # user-feature matrix
        user_x = self.m_user_embedding(user_ids)
        user_x0 = self.m_H1(user_ids)
        user_x1 = self.dropout(self.m_tag_user_embedding(user_x))
        pos_tag_input = pos_tag_input

        #user_attr_p = torch.sum(user_x1, dim=1)
        user_attr_p = user_x1
        user_attr_t = pos_tag_input


        # item-feature matrix
        item_x = self.m_item_embedding(item_ids)
        item_x0 = self.m_H2(item_ids)
        item_x1 = self.dropout(self.m_tag_item_embedding(item_x))

        #item_attr_p = torch.sum(item_x1, dim=1)
        item_attr_p = item_x1
        item_attr_t = neg_tag_input


        user_x2 = torch.cat([user_x, user_x0], dim=1)
        item_x2 = torch.cat([item_x, item_x0], dim=1)
        rating_2 = torch.mul(user_x2, item_x2)
        rating_2 = torch.sum(rating_2, dim=1)
        rating_l = rating_2 - rating

        return user_attr_p, user_attr_t, item_attr_p, item_attr_t, rating_l