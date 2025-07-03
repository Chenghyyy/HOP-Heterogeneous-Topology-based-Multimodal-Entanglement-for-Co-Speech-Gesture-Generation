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
    # epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    speaker_model = checkpoint['speaker_model']
    pose_dim = checkpoint['pose_dim']

    generator, discriminator = init_model(args, lang_model, speaker_model, pose_dim, _device)
    generator.load_state_dict(checkpoint['gen_dict'])

    # set to eval mode
    generator.train(False)

    return args, generator, lang_model, speaker_model, pose_dim

