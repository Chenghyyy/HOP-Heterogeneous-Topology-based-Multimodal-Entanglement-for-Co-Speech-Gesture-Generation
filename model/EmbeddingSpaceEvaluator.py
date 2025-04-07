import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils import weight_norm
import numpy as np
from scipy import linalg

from model.motion_ae import MotionAE


def ConvNormRelu(in_channels, out_channels, downsample=False, padding=0, batchnorm=True):
    if not downsample:
        k = 3
        s = 1
    else:
        k = 4
        s = 2

    conv_block = nn.Conv1d(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
    norm_block = nn.BatchNorm1d(out_channels)

    if batchnorm:
        net = nn.Sequential(
            conv_block,
            norm_block,
            nn.LeakyReLU(0.2, True)
        )
    else:
        net = nn.Sequential(
            conv_block,
            nn.LeakyReLU(0.2, True)
        )

    return net

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


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
        #print('out',out.shape)#(1,32,34)
        return out.transpose(1, 2)  # to (batch x seq x dim)只是做一个转置处理，但并没有改变数据储存顺序和结构

class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()

        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=args.freeze_wordembed)#freeze_wordembed=none表示可以更新嵌入
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        # print('hidden_size',args.hidden_size)
        # print('n_layers',args.n_layers)
        num_channels = [args.hidden_size] * args.n_layers#[4]*300
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)#num_channels 列表定义了每一层的输出通道数，等于 hidden_size，共有 n_layers 层。

        self.decoder = nn.Linear(num_channels[-1], 32)#将TCN的输出特征表示映射到32维的向量
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



class ContextEncoder(nn.Module):
    def __init__(self, args, n_frames, n_words, word_embed_size, word_embeddings):
        super().__init__()

        # encoders
        self.text_encoder = TextEncoderTCN(args, n_words, word_embed_size, pre_trained_embedding=word_embeddings)
        self.audio_encoder = WavEncoder()
        self.gru = nn.GRU(32+32, hidden_size=256, num_layers=2,
                          bidirectional=False, batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32)
        )

        self.fc_mu = nn.Linear(32, 32)
        self.fc_logvar = nn.Linear(32, 32)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, in_text, in_spec):
        if self.do_flatten_parameters:
            self.gru.flatten_parameters()
        text_feat_seq, _ = self.text_encoder(in_text)
        audio_feat_seq = self.audio_encoder(in_spec)

        input = torch.cat((audio_feat_seq, text_feat_seq), dim=2)
        output, _ = self.gru(input)

        last_output = output[:, -1]
        out = self.out(last_output)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        z = reparameterize(mu, logvar)
        return z, mu, logvar



class PoseEncoderConv(nn.Module):
    def __init__(self, length, dim):
        super().__init__()

        self.net = nn.Sequential(
            ConvNormRelu(dim, 32, batchnorm=True),
            ConvNormRelu(32, 64, batchnorm=True),
            ConvNormRelu(64, 64, True, batchnorm=True),
            nn.Conv1d(64, 32, 3)
        )

        self.out_net = nn.Sequential(
            # nn.Linear(864, 256),  # for 64 frames
            nn.Linear(384, 256),  # for 34 frames
            nn.BatchNorm1d(256),
            nn.LeakyReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(True),
            nn.Linear(128, 32),
        )

        self.fc_mu = nn.Linear(32, 32)
        self.fc_logvar = nn.Linear(32, 32)

    def forward(self, poses, variational_encoding):
        # encode
        #print(poses.shape)
        #print(type(poses))
        poses = poses.transpose(1, 2)  # to (bs, dim, seq)
        # print(poses.dtype)
        out = self.net(poses)
        out = out.flatten(1)
        out = self.out_net(out)

        # return out, None, None
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)

        if variational_encoding:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar

class PoseDecoderGRU(nn.Module):
    def __init__(self, gen_length, pose_dim):
        super().__init__()
        self.gen_length = gen_length
        self.pose_dim = pose_dim
        self.in_size = 32 + 32
        self.hidden_size = 300

        self.pre_pose_net = nn.Sequential(
            nn.Linear(pose_dim * 4, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.gru = nn.GRU(self.in_size, hidden_size=self.hidden_size, num_layers=4, batch_first=True,
                          bidirectional=True, dropout=0.3)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size // 2, pose_dim)
        )

    def forward(self, latent_code, pre_poses):
        pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
        feat = torch.cat((pre_pose_feat, latent_code), dim=1)
        feat = feat.unsqueeze(1).repeat(1, self.gen_length, 1)

        output, decoder_hidden = self.gru(feat)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # sum bidirectional outputs
        output = self.out(output.reshape(-1, output.shape[2]))
        output = output.view(pre_poses.shape[0], self.gen_length, -1)

        return output


class PoseDecoderConv(nn.Module):
    def __init__(self, length, dim, use_pre_poses=False):
        super().__init__()
        self.use_pre_poses = use_pre_poses

        feat_size = 32
        if use_pre_poses:
            self.pre_pose_net = nn.Sequential(
                nn.Linear(dim * 4, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 32),
            )
            feat_size += 32

        if length == 64:
            self.pre_net = nn.Sequential(
                nn.Linear(feat_size, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(True),
                nn.Linear(128, 256),
            )
        elif length == 34:
            self.pre_net = nn.Sequential(
                nn.Linear(feat_size, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(True),
                nn.Linear(64, 136),
            )
        else:
            assert False

        self.net = nn.Sequential(
            nn.ConvTranspose1d(4, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 32, 3),
            nn.Conv1d(32, dim, 3),
        )

    def forward(self, feat, pre_poses=None):
        if self.use_pre_poses:
            pre_pose_feat = self.pre_pose_net(pre_poses.reshape(pre_poses.shape[0], -1))
            feat = torch.cat((pre_pose_feat, feat), dim=1)

        out = self.pre_net(feat)
        out = out.view(feat.shape[0], 4, -1)
        out = self.net(out)
        out = out.transpose(1, 2)
        return out


class EmbeddingNet(nn.Module):
    def __init__(self, args, pose_dim, n_frames, n_words, word_embed_size, word_embeddings, mode):
        super().__init__()
        if mode != 'pose':
            self.context_encoder = ContextEncoder(args, n_frames, n_words, word_embed_size, word_embeddings)
            self.pose_encoder = PoseEncoderConv(n_frames, pose_dim)
            # self.decoder = PoseDecoderFC(n_frames, pose_dim, use_pre_poses=True)
            self.decoder = PoseDecoderGRU(n_frames, pose_dim)
        else:
            self.context_encoder = None
            self.pose_encoder = PoseEncoderConv(n_frames, pose_dim)
            self.decoder = PoseDecoderConv(n_frames, pose_dim)
        self.mode = mode

    def forward(self, in_text, in_audio, pre_poses, poses, input_mode=None, variational_encoding=False):
        if input_mode is None:
            assert self.mode is not None
            input_mode = self.mode

        # context
        if self.context_encoder is not None and in_text is not None and in_audio is not None:
            context_feat, context_mu, context_logvar = self.context_encoder(in_text, in_audio)
            # context_feat = F.normalize(context_feat, p=2, dim=1)
        else:
            context_feat = context_mu = context_logvar = None

        # poses
        if poses is not None:
            poses_feat, pose_mu, pose_logvar = self.pose_encoder(poses, variational_encoding)
            # poses_feat = F.normalize(poses_feat, p=2, dim=1)
        else:
            poses_feat = pose_mu = pose_logvar = None

        # decoder
        if input_mode == 'random':
            input_mode = 'speech' if random.random() > 0.5 else 'pose'

        if input_mode == 'speech':
            latent_feat = context_feat
        elif input_mode == 'pose':
            latent_feat = poses_feat
        else:
            assert False

        out_poses = self.decoder(latent_feat, pre_poses)

        return context_feat, context_mu, context_logvar, poses_feat, pose_mu, pose_logvar, out_poses


class EmbeddingSpaceEvaluator:
    def __init__(self, args, embed_net_path, word_embeddings, vocab_size, device):
        self.n_pre_poses = args.n_pre_poses
        # print('self.n_pre_poses',self.n_pre_poses)#4

        # init embed net
        ckpt = torch.load(embed_net_path, map_location=device)
        # print(ckpt.keys())
        # print(ckpt['pose_dim'])
        # print(ckpt['motion_ae'])
        n_frames = args.n_poses
        # print('n_frame',n_frames)
        # print('args.wordembed_dim',args.wordembed_dim)#300
        # word_embeddings = lang_model.word_embedding_weights
        mode = 'pose'
        self.pose_dim = ckpt['pose_dim']
        # print(self.pose_dim)
        # print('lang_model',lang_model.n_words)#29460
        # self.net = EmbeddingNet(args, self.pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
        #                         word_embeddings, mode).to(device)
        if args.pose_dim == 27:
            self.net = EmbeddingNet(args, self.pose_dim, n_frames, vocab_size, args.wordembed_dim,
                                    word_embeddings, mode).to(device)
            self.net.load_state_dict(ckpt['gen_dict'])
        elif args.pose_dim == 126:
            self.latent_dim = ckpt['latent_dim']
            self.net = MotionAE(self.pose_dim, self.latent_dim).to(device)
            self.net.load_state_dict(ckpt['motion_ae'])

        self.net.train(False)

        # storage
        self.context_feat_list = []
        self.real_feat_list = []
        self.generated_feat_list = []
        self.recon_err_diff = []

        self.cos_err_diff = []
        self.datasets = args.datasets

    def reset(self):
        self.context_feat_list = []
        self.real_feat_list = []
        self.generated_feat_list = []
        self.recon_err_diff = []

        self.cos_err_diff = []

    def get_no_of_samples(self):
        return len(self.real_feat_list)

    def push_samples(self, context_text, context_spec, generated_poses, real_poses):
        # convert poses to latent features
        if self.pose_dim == 27:
            pre_poses = real_poses[:, 0:self.n_pre_poses]
            context_feat, _, _, real_feat, _, _, real_recon = self.net(context_text, context_spec, pre_poses, real_poses,
                                                                       'pose', variational_encoding=False)
            _, _, _, generated_feat, _, _, generated_recon = self.net(None, None, pre_poses, generated_poses,
                                                                      'pose', variational_encoding=False)

            # print('real_feat',real_feat.shape)#[32, 32]
            # print('generated_feat',generated_feat.shape)#[32, 32]
            if context_feat:
                self.context_feat_list.append(context_feat.data.cpu().numpy())
        elif self.pose_dim == 126:
            # print(123)
            real_recon, real_feat = self.net(real_poses)
            generated_recon, generated_feat = self.net(generated_poses)

        # print('real_feat',real_feat)
        # print('generated_feat',generated_feat)
        self.real_feat_list.append(real_feat.data.cpu().numpy())
        self.generated_feat_list.append(generated_feat.data.cpu().numpy())

        # reconstruction error
        if self.datasets == 'TED':
            recon_err_real = F.l1_loss(real_poses, real_recon).item()
            recon_err_fake = F.l1_loss(generated_poses, generated_recon).item()
            self.recon_err_diff.append(recon_err_fake - recon_err_real)

        else:
            reconstruction_loss_real = F.l1_loss(real_recon, real_poses, reduction='none')
            reconstruction_loss_real = torch.mean(reconstruction_loss_real, dim=(1, 2))

            if True:  # use pose diff
                target_diff = real_poses[:, 1:] - real_poses[:, :-1]
                recon_diff = real_recon[:, 1:] - real_recon[:, :-1]
                reconstruction_loss_real += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

            reconstruction_loss_real = torch.sum(reconstruction_loss_real)
            cos_loss_real = torch.sum(
                1 - torch.cosine_similarity(real_recon.view(real_recon.shape[0], real_recon.shape[1], -1, 3),
                                            real_poses.view(real_poses.shape[0], real_poses.shape[1], -1, 3), dim=-1))

            reconstruction_loss_fake = F.l1_loss(generated_recon, generated_poses, reduction='none')
            reconstruction_loss_fake = torch.mean(reconstruction_loss_fake, dim=(1, 2))

            if True:  # use pose diff
                target_diff = generated_poses[:, 1:] - generated_poses[:, :-1]
                recon_diff = generated_recon[:, 1:] - generated_recon[:, :-1]
                reconstruction_loss_fake += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

            reconstruction_loss_fake = torch.sum(reconstruction_loss_fake)
            cos_loss_fake = torch.sum(
                1 - torch.cosine_similarity(generated_recon.view(generated_recon.shape[0], generated_recon.shape[1], -1, 3),
                                            generated_poses.view(generated_poses.shape[0], generated_poses.shape[1], -1, 3),
                                            dim=-1))

            self.recon_err_diff.append(reconstruction_loss_fake - reconstruction_loss_real)
            self.cos_err_diff.append(cos_loss_fake - cos_loss_real)

    def get_diversity_scores(self):
        feat1 = np.vstack(self.generated_feat_list[:500])
        random_idx = torch.randperm(len(self.generated_feat_list))[:500]
        shuffle_list = [self.generated_feat_list[x] for x in random_idx]
        feat2 = np.vstack(shuffle_list)
        # dists = []
        # for i in range(feat1.shape[0]):
        #     d = np.sum(np.absolute(feat1[i] - feat2[i]))  # MAE
        #     dists.append(d)
        feat_dist = np.mean(np.sum(np.absolute(feat1 - feat2), axis=-1))
        return feat_dist

    def get_scores(self):
        generated_feats = np.vstack(self.generated_feat_list)
        # print('generated_feats_list',generated_feats.shape)#(26880, 32)
        real_feats = np.vstack(self.real_feat_list)
        # print('real_feats_list',real_feats.shape)#(26880, 32)

        def frechet_distance(samples_A, samples_B):
            A_mu = np.mean(samples_A, axis=0)
            A_sigma = np.cov(samples_A, rowvar=False)
            B_mu = np.mean(samples_B, axis=0)
            B_sigma = np.cov(samples_B, rowvar=False)
            try:
                frechet_dist = self.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
            except ValueError:
                frechet_dist = 1e+10
            return frechet_dist

        ####################################################################
        # frechet distance
        frechet_dist = frechet_distance(generated_feats, real_feats)

        ####################################################################
        # distance between real and generated samples on the latent feature space
        dists = []
        for i in range(real_feats.shape[0]):
            d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
            dists.append(d)
        feat_dist = np.mean(dists)

        return frechet_dist, feat_dist

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
