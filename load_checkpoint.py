from model.multimodal_context_net import PoseGenerator, ConvDiscriminator
import torch

def init_model(args, lang_model, speaker_model, pose_dim, _device):
    print(_device)
    # init model
    n_frames = args.n_poses
    generator = None
    if args.model == 'multimodal_context':
        generator = PoseGenerator(args,
                                  n_words=lang_model.n_words,
                                  word_embed_size=args.wordembed_dim,
                                  word_embeddings=lang_model.word_embedding_weights,
                                  z_obj=speaker_model,
                                  pose_dim=pose_dim).to(_device)
        discriminator = ConvDiscriminator(pose_dim).to(_device)
    return generator, discriminator

def load_checkpoint_and_model(checkpoint_path, _device='cpu'):
    print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    speaker_model = checkpoint['speaker_model']
    pose_dim = checkpoint['pose_dim']
    #print(type(checkpoint['gen_dict']))
    #print(checkpoint['gen_dict'])
    # print('epoch {}'.format(epoch))

    # 要删除的键列表
    # keys_to_delete = ['speaker_embedding.0.weight','speaker_embedding.1.weight','speaker_embedding.1.bias','speaker_mu.weight','speaker_mu.bias','speaker_logvar.weight','speaker_logvar.bias']
    #
    # # 使用循环删除指定的键
    # for key in keys_to_delete:
    #     if key in checkpoint['gen_dict']:
    #         del checkpoint['gen_dict'][key]

    generator, discriminator = init_model(args, lang_model, speaker_model, pose_dim, _device)
    generator.load_state_dict(checkpoint['gen_dict'])

    # set to eval mode
    generator.train(False)

    return args, generator, lang_model, speaker_model, pose_dim

