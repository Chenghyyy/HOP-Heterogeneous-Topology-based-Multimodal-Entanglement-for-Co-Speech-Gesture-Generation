import torch.nn.functional as F
from data_loader.lmdb_data_loader import *


def add_noise(data):
    noise = torch.randn_like(data) * 0.1
    return data + noise

def train_llm(args, epoch, in_audio, log_melspec, text_token_padded, target_dir_vec, vid_indices,
              model, discriminator,
              model_optim, dis_optimizer, accelerator):
    pre_seq = target_dir_vec[:, 0:16]
    use_noisy_target = True
    dis_error = None
    if epoch > 10 and args.loss_gan_weight > 0.0:
        dis_optimizer.zero_grad()
        outputs, *_ = model(in_audio, log_melspec, text_token_padded, pre_seq, vid_indices)
        # target_dir_vec = target_dir_vec.to(torch.float32)
        # print('target_dir_vec.shape',target_dir_vec.shape)

        # #####训练判别器
        if use_noisy_target:
            noise_target = add_noise(target_dir_vec)
            noise_out = add_noise(outputs.detach())
            dis_real = discriminator(noise_target, text_token_padded)
            dis_fake = discriminator(noise_out, text_token_padded)
        else:
            dis_real = discriminator(target_dir_vec, text_token_padded)
            dis_fake = discriminator(outputs.detach(), text_token_padded)

        dis_error = torch.sum(-torch.mean(torch.log(dis_real + 1e-8) + torch.log(1 - dis_fake + 1e-8)))  # ns-gan
        # dis_loss.append(dis_error.item())
        #
        accelerator.backward(dis_error)
        # dis_error.backward()
        dis_optimizer.step()
    # #####

    #####训练生成器
    model_optim.zero_grad()

    outputs, z_context, z_mu, z_logvar = model(in_audio, log_melspec, text_token_padded, pre_seq, vid_indices)
    dis_output = discriminator(outputs, text_token_padded)
    gen_error = -torch.mean(torch.log(dis_output + 1e-8))

    huber_loss = F.smooth_l1_loss(outputs / 0.1, target_dir_vec / 0.1) * 0.1
    # print('loss',loss)
    kld = div_reg = None

    if (args.z_type == 'speaker' or args.z_type == 'random') and args.loss_reg_weight > 0.0:
        if args.z_type == 'speaker':
            # enforcing divergent gestures btw original vid and other vid
            rand_idx = torch.randperm(vid_indices.shape[0])
            rand_vids = vid_indices[rand_idx]
        else:
            rand_vids = None

        out_dir_vec_rand_vid, z_rand_vid, _, _ = model(in_audio, log_melspec, text_token_padded, pre_seq, rand_vids)
        beta = 0.05
        pose_l1 = F.smooth_l1_loss(outputs / beta, out_dir_vec_rand_vid.detach() / beta, reduction='none') * beta
        # print('pose_l1',pose_l1)
        pose_l1 = pose_l1.sum(dim=1).sum(dim=1)

        pose_l1 = pose_l1.view(pose_l1.shape[0], -1).mean(1)
        z_l1 = F.l1_loss(z_context.detach(), z_rand_vid.detach(), reduction='none')
        z_l1 = z_l1.view(z_l1.shape[0], -1).mean(1)
        div_reg = -(pose_l1 / (z_l1 + 1.0e-5))
        div_reg = torch.clamp(div_reg, min=-1000)
        div_reg = div_reg.mean()

        if args.z_type == 'speaker':
            # speaker embedding KLD
            kld = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
            loss = huber_loss * args.loss_regression_weight + div_reg * args.loss_reg_weight + kld * args.loss_kld_weight
            # 600,0.4,0.6
        else:
            loss = huber_loss * args.loss_regression_weight + div_reg * args.loss_reg_weight
    else:
        loss = huber_loss * args.loss_regression_weight  # + var_loss

    if epoch > 10:
        loss += gen_error * args.loss_gan_weight
    # train_loss.append(loss.item())

    accelerator.backward(loss)  # 反向传播
    model_optim.step()

    ret_dict = {'loss': args.loss_regression_weight * huber_loss.item()}
    if kld:
        ret_dict['KLD'] = args.loss_kld_weight * kld.item()
    if div_reg:
        ret_dict['DIV_REG'] = args.loss_reg_weight * div_reg.item()

    if epoch > 10 and args.loss_gan_weight > 0.0:
        ret_dict['gen'] = args.loss_gan_weight * gen_error.item()
        ret_dict['dis'] = dis_error.item()

    return ret_dict