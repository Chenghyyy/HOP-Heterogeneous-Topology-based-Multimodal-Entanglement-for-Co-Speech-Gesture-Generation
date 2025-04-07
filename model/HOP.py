from math import sqrt
import transformers
from  model import embedding_net
import torch
import torch.nn as nn
from model import gwnet
from model.EmbeddingSpaceEvaluator import TemporalConvNet

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()

class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()

        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=False)
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        num_channels = [300] * 4
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], 32)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0

class WavEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=5, padding=1600),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(16, 32, 15, stride=6),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(32, 64, 15, stride=6),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv1d(64, 32, 15, stride=6),
        )

    def forward(self, wav_data):
        wav_data = wav_data.unsqueeze(1)  # add channel dim
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)  # to (batch x seq x dim)只是做一个转置处理，但并没有改变数据储存顺序和结构


class Model(nn.Module):
    def __init__(self, configs, model, tokenizer, z_obj=None):
        super(Model, self).__init__()
        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        self.llm_model = model
        self.tokenizer = tokenizer
        self.z_obj = z_obj
        self.use_gwnet = configs.use_gwnet
        self.use_reprograme = configs.use_reprograme

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.audio_encoder = WavEncoder()

        #####
        self.speaker_embedding = None
        if self.z_obj:
            # print(123)
            self.z_size = 16
            # self.in_size += self.z_size
            # if isinstance(self.z_obj, vocab.Vocab):
            self.speaker_embedding = nn.Sequential(
                nn.Embedding(z_obj.n_words, self.z_size),
                nn.Linear(self.z_size, self.z_size)
            )
            self.speaker_mu = nn.Linear(self.z_size, self.z_size)
            self.speaker_logvar = nn.Linear(self.z_size, self.z_size)
            # else:
            #     pass  # random noise

        self.word_embeddings = self.llm_model.get_input_embeddings().weight#.index_select(dim=0, index=self.dim0)
        # self.word_embeddings = self.word_embeddings.index_select(dim=1, index=self.dim1)
        self.vocab_size = self.word_embeddings.shape[0]#self.word_embeddings.shape[0]#60
        if self.use_reprograme:
            self.num_tokens = 1500
            self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

            self.align_layer = nn.Linear(2 * self.d_llm, self.d_llm)
            self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        if configs.datasets == 'TED':
            self.pred_g_len = 27
        else:
            self.pred_g_len = 126
        self.hidden_size = 350
        # self.gru_input_size = self.d_llm + 27 + 1 + 16 + 180

        #############
        if self.use_gwnet:
            self.beat = nn.Sequential(
                nn.Linear(3400, 1700),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Linear(1700, 170)
            )

            if configs.datasets == 'TED':
                num_nodes = 9
            else:
                num_nodes = 42

            in_dim = 173#in_channels
            out_dim = 173
            self.gwnet = gwnet.gwnet(device, num_nodes, dropout=0, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=in_dim, out_dim=out_dim, residual_channels=64, dilation_channels=64, skip_channels=256, end_channels=512)

        #############
        if configs.datasets == 'TED':
            if self.use_reprograme and self.use_gwnet:
                self.gru_input_size = self.d_llm + 27 + 1 + 16 + 180
            elif self.use_reprograme and self.use_gwnet is False:
                self.gru_input_size = self.d_llm + 27 + 1 + 16 + 32
            elif self.use_reprograme is False and self.use_gwnet:
                self.gru_input_size = self.d_llm + 27 + 1 + 16 + 180
            elif self.use_reprograme is False and self.use_gwnet is False:
                self.gru_input_size = self.d_llm + 27 + 1 + 16 + 32
        else:
            if self.use_reprograme and self.use_gwnet:
                self.gru_input_size = self.d_llm + 126 + 1 + 16 + 840
            elif self.use_reprograme and self.use_gwnet is False:
                self.gru_input_size = self.d_llm + 126 + 1 + 16 + 32
            elif self.use_reprograme is False and self.use_gwnet:
                self.gru_input_size = self.d_llm + 126 + 1 + 16 + 840
            elif self.use_reprograme is False and self.use_gwnet is False:
                self.gru_input_size = self.d_llm + 126 + 1 + 16 + 32

        # self.flatten = nn.Flatten(start_dim=-2)
        self.gru = nn.GRU(self.gru_input_size, hidden_size=self.hidden_size, num_layers=4, batch_first=True,#传入gru前可展平输入，因为这个输入有时间依赖的在第二维，展平后就都在一维了，也好做输出的大小
                          bidirectional=True, dropout=0).to(torch.float32).to(device)

        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Dropout(0),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size // 2, self.pred_g_len)
        )
        # self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, in_audio, x_enc, text, pre_seq, vid_indices= None):
        dec_out = self.forecast(in_audio, x_enc, text, pre_seq, vid_indices)
        return dec_out

    def forecast(self, in_audio, x_enc, text, pre_seq, vid_indices):
        num_joints = int(pre_seq.shape[2] / 3)

        if self.z_obj:
            if self.speaker_embedding:
                assert vid_indices is not None
                z_context = self.speaker_embedding(vid_indices)
                z_mu = self.speaker_mu(z_context)
                z_logvar = self.speaker_logvar(z_context)
                z_context = embedding_net.reparameterize(z_mu, z_logvar)
            else:
                z_mu = z_logvar = None
                z_context = torch.randn(text.shape[0], self.z_size, device=x_enc.device)
        else:
            z_mu = z_logvar = None
            z_context = None

        text_embeddings = self.llm_model.get_input_embeddings()(text.to(x_enc.device).long())  # (batch, prompt_token, dim)
        if self.use_reprograme:
            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
            enc_out = self.reprogramming_layer(x_enc, source_embeddings, source_embeddings)
            llama_enc_out = torch.cat([enc_out, text_embeddings], dim=2)
            llama_enc_out = self.align_layer(llama_enc_out)
            dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        else:
            dec_out = self.llm_model(inputs_embeds=text_embeddings).last_hidden_state


        if self.use_gwnet:
            in_audio = in_audio.unfold(1, 3400, 2191).unsqueeze(1).repeat(1, num_joints, 1, 1)
            in_audio = self.beat(in_audio)
            in_audio = in_audio.view(in_audio.shape[0], 16, num_joints, 170)

            pre_seq = pre_seq.view(pre_seq.shape[0], 16, -1, 3)  # reshape: (batch_size, 4, 9, 3)

            seq_audio = torch.cat([pre_seq, in_audio], dim=3)
            seq_audio = seq_audio.permute(0, 3, 2, 1)

            feature = self.gwnet(seq_audio)

            g_seq = feature[:, :3, :, :]
            beat = feature[:, 3:, :, :]
            beat = beat.view(beat.shape[0], 34, -1)

            g_seq = g_seq.view(g_seq.shape[0], -1, g_seq.shape[3])
            g_seq = g_seq.permute(0, 2, 1)
            pre_seq = g_seq.new_zeros((g_seq.shape[0], 34, g_seq.shape[2] + 1))
            pre_seq[:, 0:g_seq.shape[1], :-1] = g_seq[:, 0:g_seq.shape[1]]
            pre_seq[:, 0:g_seq.shape[1], -1] = 1

            dec_out = torch.cat([pre_seq, beat, dec_out], dim=2)
        else:
            ges_seq = pre_seq.new_zeros((pre_seq.shape[0], 34, pre_seq.shape[2] + 1))
            ges_seq[:, 0:pre_seq.shape[1], :-1] = pre_seq[:, 0:pre_seq.shape[1]]
            ges_seq[:, 0:pre_seq.shape[1], -1] = 1

            audio_feature = self.audio_encoder(in_audio)

            dec_out = torch.cat([ges_seq, audio_feature, dec_out], dim=2)

        if z_context is not None:
            # print('z_context.shape',z_context.shape)#torch.Size([128, 16])
            repeated_z = z_context.unsqueeze(1)
            repeated_z = repeated_z.repeat(1, 34, 1)
            # print('repeated_z',repeated_z.shape)#[128, 34, 16]
            dec_out = torch.cat([dec_out, repeated_z], dim=2)

        dec_out, decoder_hidden = self.gru(dec_out.to(torch.float32).contiguous(), None)
        dec_out = dec_out[:, :, :self.hidden_size] + dec_out[:, :, self.hidden_size:]
        dec_out = self.out(dec_out)
        #dec_out = self.normalize_layers(dec_out, 'denorm')
        return dec_out, z_context, z_mu, z_logvar


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        #d_model=32,d_keys=128,n_heads=8，d_llm=768
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)#（32，1024）
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)#（768，1024）
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)#（768，1024）
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)#（1024，768）
        self.n_heads = n_heads
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(attention_dropout)

    #target_embedding=enc_out=(1,64,32),source_embedding=value_embedding=(1000,768)
    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        out = self.activation(out)
        return self.out_projection(out)#表示给重新变成为语言的输出连一个全连接层，做完一个attention后一般需要一个全连接层
        #全连接层可以根据输入数据的维度大小控制输出的维度的大小
        #(4,64,768)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)#cross attention操作

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding#（4，64，8，128）
