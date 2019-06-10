import torch
import torch.nn as nn


class getEmbeddings(nn.Module):
    def __init__(self, word_size, word_length, feature_size, feature_length, Wv, pf1, pf2):
        super(getEmbeddings, self).__init__()
        self.x_embedding = nn.Embedding(word_length, word_size, padding_idx=0)  # 当前词的embedding
        self.ldist_embedding = nn.Embedding(feature_length, feature_size, padding_idx=0)  # 当前词与entity1之间的相对距离embedding
        self.rdist_embedding = nn.Embedding(feature_length, feature_size, padding_idx=0)  # 当前词与entity2之间的相对距离embedding
        self.x_embedding.weight.data.copy_(torch.from_numpy(Wv))
        self.ldist_embedding.weight.data.copy_(torch.from_numpy(pf1))
        self.rdist_embedding.weight.data.copy_(torch.from_numpy(pf2))

    def forward(self, x, ldist, rdist):
        x_embed = self.x_embedding(x)
        ldist_embed = self.ldist_embedding(ldist)
        rdist_embed = self.rdist_embedding(rdist)
        concat = torch.cat([x_embed, ldist_embed, rdist_embed], x_embed.dim() - 1)
        return concat.unsqueeze(1)  # 在第一维上增加一个维度

    # word_size: 词emb维度
    # word_length: 词个数
