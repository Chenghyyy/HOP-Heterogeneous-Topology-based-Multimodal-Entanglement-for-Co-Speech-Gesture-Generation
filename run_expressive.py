import argparse
import time
import math

import numpy as np
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from tqdm import tqdm
# from history import TimeLLM_mae_copy_copy
from model import HOP
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content
from model.EmbeddingSpaceEvaluator import EmbeddingSpaceEvaluator
from load_checkpoint import load_checkpoint_and_model
from model.multimodal_context_net import PoseGenerator, ConvDiscriminator
from model.hierarchy_net import Hierarchical_PoseGenerator, Hierarchical_ConvDiscriminator, Hierarchical_WavEncoder, TextEncoderTCN

from data_loader.lmdb_data_loader_expressive import *
# from data_loader.lmdb_data_loader import *
from Evaluate import evaluate_testset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from utils.vocab_utils import build_vocab

from utils.average_meter import AverageMeter
from convert import convert_dir_vec_to_pose

from train_eval.train_llm import train_llm
from train_eval.train_gan import train_iter_gan
from train_eval.train_joint_embed import train_iter_embed, eval_embed
from train_eval.train_seq2seq import train_iter_seq2seq
from train_eval.train_speech2gesture import train_iter_speech2gesture

from model.embedding_net import EmbeddingNet
from model.seq2seq_net import Seq2SeqNet
from model import speech2gesture, vocab

#, train_gan, train_hierarchy, train_joint_embed, train_seq2seq, train_speech2gesture

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

parser = argparse.ArgumentParser(description='Time-LLM')

# vid_idx = random.sample(range(0, 1000), 1)[0]

#为了确保程序中的随机性操作在每次运行时产生相同的结果，即“固定随机种子”。这样做可以使结果可复现，特别是在调试和实验过程中
# fix_seed = 2021
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)

# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model', type=str, required=False, default='AD_LLM',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# model define
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--llm_model', type=str, default='BERT', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

#gesture
parser.add_argument('--datasets', type=str, default='TED_expressive',help='options: [TED, TED_expressive]')
parser.add_argument('--n_poses', type=int, default=34)
parser.add_argument('--pose_dim', type=int, default=126)
parser.add_argument('--wordembed_dim', type=int, default=300)
parser.add_argument('--n_pre_poses', type=int, default=4)
parser.add_argument("--z_type", type=str, default='speaker')

parser.add_argument("--loss_regression_weight", type=float, default=2100)
parser.add_argument("--loss_gan_weight", type=float, default=5)
parser.add_argument("--loss_kld_weight", type=float, default=0.8)
parser.add_argument("--loss_reg_weight", type=float, default=0.5)
parser.add_argument('--generator',type=str, default='LLM_generator',
                    help='model name, options: [hierarchy, multimodal_context, joint_embedding,gesture_autoencoder,seq2seq, speech2gesture]')
parser.add_argument("--use_gwnet", default=True)
parser.add_argument('--use_reprograme', default=True)

# optimization
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=75, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

angle_pair = [
    (0, 1),
    (0, 2),
    (1, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (8, 9),
    (9, 10),
    (11, 12),
    (12, 13),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (17, 5),
    (5, 8),
    (8, 14),
    (14, 11),
    (2, 20),
    (20, 21),
    (22, 23),
    (23, 24),
    (25, 26),
    (26, 27),
    (28, 29),
    (29, 30),
    (31, 32),
    (32, 33),
    (34, 35),
    (35, 36),
    (34, 22),
    (22, 25),
    (25, 31),
    (31, 28),
    (0, 37),
    (37, 38),
    (37, 39),
    (38, 40),
    (39, 41),
    # palm
    (4, 42),
    (21, 43)
]

change_angle = [0.0027804733254015446, 0.002761547453701496, 0.005953566171228886, 0.013764726929366589,
    0.022748252376914024, 0.039307352155447006, 0.03733552247285843, 0.03775784373283386, 0.0485558956861496,
    0.032914578914642334, 0.03800227493047714, 0.03757007420063019, 0.027338404208421707, 0.01640886254608631,
    0.003166505601257086, 0.0017252820543944836, 0.0018696568440645933, 0.0016072227153927088, 0.005681346170604229,
    0.013287615962326527, 0.021516695618629456, 0.033936675637960434, 0.03094293735921383, 0.03378918394446373,
    0.044323261827230453, 0.034706637263298035, 0.03369896858930588, 0.03573163226246834, 0.02628341130912304,
    0.014071882702410221, 0.0029828345868736506, 0.0015706412959843874, 0.0017107439925894141, 0.0014634154504165053,
    0.004873405676335096, 0.002998138777911663, 0.0030240598134696484, 0.0009890805231407285, 0.0012279648799449205,
    0.047324635088443756, 0.04472292214632034]

#得到随机种子必须的向量
mean_dir_vec = [-0.0737964, -0.9968923, -0.1082858,  0.9111595,  0.2399522, -0.102547 , -0.8936886,  0.3131501, -0.1039348,  0.2093927, 0.958293 ,  0.0824881, -0.1689021, -0.0353824, -0.7588258, -0.2794763, -0.2495191, -0.614666 , -0.3877234,  0.005006 , -0.5301695, -0.5098616,  0.2257808,  0.0053111, -0.2393621, -0.1022204, -0.6583039, -0.4992898,  0.1228059, -0.3292085, -0.4753748,  0.2132857,  0.1742853, -0.2062069,  0.2305175, -0.5897119, -0.5452555,  0.1303197, -0.2181693, -0.5221036, 0.1211322,  0.1337591, -0.2164441,  0.0743345, -0.6464546, -0.5284583,  0.0457585, -0.319634 , -0.5074904,  0.1537192, 0.1365934, -0.4354402, -0.3836682, -0.3850554, -0.4927187, -0.2417618, -0.3054556, -0.3556116, -0.281753 , -0.5164358, -0.3064435,  0.9284261, -0.067134 ,  0.2764367,  0.006997 , -0.7365526,  0.2421269, -0.225798 , -0.6387642,  0.3788997, 0.0283412, -0.5451686,  0.5753376,  0.1935219,  0.0632555, 0.2122412, -0.0624179, -0.6755542,  0.5212831,  0.1043523, -0.345288 ,  0.5443628,  0.128029 ,  0.2073687,  0.2197118, 0.2821399, -0.580695 ,  0.573988 ,  0.0786667, -0.2133071, 0.5532452, -0.0006157,  0.1598754,  0.2093099,  0.124119, -0.6504359,  0.5465003,  0.0114155, -0.3203954,  0.5512083, 0.0489287,  0.1676814,  0.4190787, -0.4018607, -0.3912126, 0.4841548, -0.2668508, -0.3557675,  0.3416916, -0.2419564, -0.5509825,  0.0485515, -0.6343101, -0.6817347, -0.4705639, -0.6380668,  0.4641643,  0.4540192, -0.6486361,  0.4604001, -0.3256226,  0.1883097,  0.8057457,  0.3257385,  0.1292366, 0.815372]
mean_pose = [-0.0046788, -0.5397806,  0.007695 , -0.0171913, -0.7060388,-0.0107034,  0.1550734, -0.6823077, -0.0303645, -0.1514748,   -0.6819547, -0.0268262,  0.2094328, -0.469447 , -0.0096073,   -0.2318253, -0.4680838, -0.0444074,  0.1667382, -0.4643363,   -0.1895118, -0.1648597, -0.4552845, -0.2159728,  0.1387546,   -0.4859474, -0.2506667,  0.1263615, -0.4856088, -0.2675801,   0.1149031, -0.4804542, -0.267329 ,  0.1414847, -0.4727709,   -0.2583424,  0.1262482, -0.4686185, -0.2682536,  0.1150217,   -0.4633611, -0.2640182,  0.1475897, -0.4415648, -0.2438853,   0.1367996, -0.4383164, -0.248248 ,  0.1267222, -0.435534 ,   -0.2455436,  0.1455485, -0.4557491, -0.2521977,  0.1305471,   -0.4535603, -0.2611591,  0.1184687, -0.4495366, -0.257798 ,   0.1451682, -0.4802511, -0.2081622,  0.1301337, -0.4865308,   -0.2175783,  0.1208341, -0.4932623, -0.2311025, -0.1409241,-0.4742868, -0.2795303, -0.1287992, -0.4724431, -0.2963172,-0.1159225, -0.4676439, -0.2948754, -0.1427748, -0.4589126,-0.2861245, -0.126862 , -0.4547355, -0.2962466, -0.1140265,-0.451308 , -0.2913815, -0.1447202, -0.4260471, -0.2697673,-0.1333492, -0.4239912, -0.2738043, -0.1226859, -0.4238346,-0.2706725, -0.1446909, -0.440342 , -0.2789209, -0.1291436,-0.4391063, -0.2876539, -0.1160435, -0.4376317, -0.2836147,-0.1441438, -0.4729031, -0.2355619, -0.1293268, -0.4793807,-0.2468831, -0.1204146, -0.4847246, -0.2613876, -0.0056085,-0.9224338, -0.1677302, -0.0352157, -0.963936 , -0.1388849,0.0236298, -0.9650772, -0.1385154, -0.0697098, -0.9514691,-0.055632 ,  0.0568838, -0.9565502, -0.0567985]
data_mean_dir_vec = np.array(mean_dir_vec).reshape(-1, 3)

#导入别人训练好的计算FGD的模型
# checkpoint_path = '/home/wxp/chy/Gesture-Generation-from-Trimodal-Context-master/output/train_multimodal_context/multimodal_context_checkpoint_best.bin'
# eval_net_path = '/home/wxp/chy/text_evaluator/output/train_h36m_gesture_autoencoder/gesture_autoencoder_checkpoint_best.bin'
eval_net_path = '/home/wxp/chy/HOP/output/train_h36m_gesture_autoencoder/ted_expressive_autoencoder_checkpoint_best.bin'
# args_eval, generator, _, __, out_dim = load_checkpoint_and_model(checkpoint_path, device)

eval_dict = {'frechet': 1}
best_values = {'frechet': 1e+10}
# start = time.time()

if args.llm_model == 'LLAMA':
    # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
    llama_config = LlamaConfig.from_pretrained("/home/wxp/chy/HOP/llama_model/")
    llama_config.num_hidden_layers = args.llm_layers
    llama_config.output_attentions = True
    llama_config.output_hidden_states = True
    try:
        llm_model = LlamaModel.from_pretrained(
            # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
            "/home/wxp/chy/HOP/llama_model/",
            trust_remote_code=True,
            local_files_only=True,
            config=llama_config,
            # load_in_4bit=True
        )
    except EnvironmentError:  # downloads model from HF is not already done
        print("Local model files not found. Attempting to download...")
        llm_model = LlamaModel.from_pretrained(
            # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
            'huggyllama/llama-7b',
            trust_remote_code=True,
            local_files_only=False,
            config=llama_config,
            # load_in_4bit=True
        )
    try:
        tokenizer = LlamaTokenizer.from_pretrained(
            # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
            "/home/wxp/chy/HOP/llama_model/",
            # "/home/wxp/LLaMA-Factory/merge_models/llama_lora_sft/",
            trust_remote_code=True,
            local_files_only=True
        )
    except EnvironmentError:  # downloads the tokenizer from HF if not already done
        print("Local tokenizer files not found. Atempting to download them..")
        tokenizer = LlamaTokenizer.from_pretrained(
            # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
            'huggyllama/llama-7b',
            trust_remote_code=True,
            local_files_only=False
        )
elif args.llm_model == 'BERT':
    bert_config = BertConfig.from_pretrained("/home/wxp/chy/Time-LLM-main/bert_model/")
    bert_config.num_hidden_layers = args.llm_layers
    bert_config.output_attentions = True
    bert_config.output_hidden_states = True
    try:
        llm_model = BertModel.from_pretrained(
            "/home/wxp/chy/Time-LLM-main/bert_model/",
            trust_remote_code=True,
            local_files_only=True,
            config=bert_config,
        )
    except EnvironmentError:  # downloads model from HF is not already done
        print("Local model files not found. Attempting to download...")
        llm_model = BertModel.from_pretrained(
            'google-bert/bert-base-uncased',
            trust_remote_code=True,
            local_files_only=False,
            config=bert_config,
        )

    try:
        tokenizer = BertTokenizer.from_pretrained(
            "/home/wxp/chy/Time-LLM-main/bert_model/",
            trust_remote_code=True,
            local_files_only=True
        )
    except EnvironmentError:  # downloads the tokenizer from HF if not already done
        print("Local tokenizer files not found. Atempting to download them..")
        tokenizer = BertTokenizer.from_pretrained(
            'google-bert/bert-base-uncased',
            trust_remote_code=True,
            local_files_only=False
        )
else:
    raise Exception('LLM model is not defined')
llm_model = llm_model.to(device)

####################

tb_path = 'model' + '_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
tb_writer = SummaryWriter(log_dir=str(Path('/home/wxp/chy/HOP/view') / 'tensorboard_runs' / tb_path))

for ii in range(args.itr):
        train_dataset = SpeechMotionDataset('/home/wxp/data/ted_expressive_dataset/train',
                                            n_poses=34,
                                            subdivision_stride=10,
                                            pose_resampling_fps=15,
                                            mean_dir_vec=data_mean_dir_vec,
                                            mean_pose=mean_pose,
                                            tokenizer=tokenizer,
                                            remove_word_timing=('text')
                                            )

        speaker_model = train_dataset.speaker_model

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True, num_workers=4, pin_memory=True,
                                  collate_fn=default_collate_fn
                                  )

        val_dataset = SpeechMotionDataset('/home/wxp/data/ted_expressive_dataset/val',
                                          n_poses=34,
                                          subdivision_stride=10,
                                          pose_resampling_fps=15,
                                          speaker_model=speaker_model,
                                          mean_dir_vec=data_mean_dir_vec,
                                          mean_pose=mean_pose,
                                          tokenizer=tokenizer,
                                          remove_word_timing=('text')
                                          )

        test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                 shuffle=False, drop_last=True, num_workers=4, pin_memory=True,
                                 collate_fn=default_collate_fn
                                 )

        test_dataset = SpeechMotionDataset('/home/wxp/data/ted_expressive_dataset/test',
                                           n_poses=34,
                                           subdivision_stride=10,
                                           pose_resampling_fps=15,
                                           speaker_model=speaker_model,
                                           mean_dir_vec=data_mean_dir_vec,
                                           mean_pose=mean_pose,
                                           tokenizer=tokenizer)

        vocab_cache_path = '/home/wxp/data/ted_expressive_dataset/vocab_cache.pkl'
        wordembed_path = '/home/wxp/chy/HOP/data/fasttext/crawl-300d-2M-subword/crawl-300d-2M-subword.bin'
        lang_model = build_vocab('words', [train_dataset, val_dataset, test_dataset], vocab_cache_path, wordembed_path, 300)
        train_dataset.set_lang_model(lang_model)
        val_dataset.set_lang_model(lang_model)

        pose_dim = args.pose_dim
        n_frames = 34

        if args.model == 'AD_LLM':
            model = HOP.Model(args, llm_model, tokenizer, speaker_model).float()
            model.to(torch.float32)  # 加gru后
            discriminator = ConvDiscriminator(pose_dim).to(device)
        if args.model == 'hierarchy':
            generator = Hierarchical_PoseGenerator(args,
                                                   n_words=lang_model.n_words,
                                                   word_embed_size=args.wordembed_dim,
                                                   word_embeddings=lang_model.word_embedding_weights,
                                                   z_obj=speaker_model,
                                                   pose_dim=pose_dim)
            discriminator = Hierarchical_ConvDiscriminator(pose_dim)
            audio_encoder = Hierarchical_WavEncoder(args, z_obj=speaker_model, pose_level=3, nOut=32)
            text_encoder = TextEncoderTCN(args, lang_model.n_words, args.wordembed_dim,
                                          pre_trained_embedding=lang_model.word_embedding_weights,
                                          dropout=args.dropout_prob)
        elif args.model == 'multimodal_context':
            generator = PoseGenerator(args,
                                      n_words=lang_model.n_words,
                                      word_embed_size=args.wordembed_dim,
                                      word_embeddings=lang_model.word_embedding_weights,
                                      z_obj=speaker_model,
                                      pose_dim=pose_dim)
            discriminator = ConvDiscriminator(pose_dim)
        elif args.model == 'joint_embedding':
            generator = EmbeddingNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                                     lang_model.word_embedding_weights, mode='random')
        elif args.model == 'gesture_autoencoder':
            generator = EmbeddingNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                                     lang_model.word_embedding_weights, mode='pose')
        elif args.model == 'seq2seq':
            generator = Seq2SeqNet(args, pose_dim, n_frames, lang_model.n_words, args.wordembed_dim,
                                   lang_model.word_embedding_weights)
            loss_fn = torch.nn.L1Loss()
        elif args.model == 'speech2gesture':
            generator = speech2gesture.Generator(n_frames, pose_dim, args.n_pre_poses)
            discriminator = speech2gesture.Discriminator(pose_dim)
            loss_fn = torch.nn.L1Loss()

        time_now = time.time()
        train_steps = len(train_loader)
        print('train_steps',train_steps)

        if args.model == 'hierarchy':

            # gen_optimizer = optim.Adam(list(g1.parameters()) +
            #                             list(g2.parameters()) +
            #                              list(g3.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
            gen_optimizer_1 = optim.Adam(g1.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
            gen_optimizer_2 = optim.Adam(g2.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
            gen_optimizer_3 = optim.Adam(g3.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
            audio_optimizer = optim.Adam(audio_encoder.parameters(),
                                         lr=args.learning_rate,
                                         betas=(0.5, 0.999))
            text_optimizer = optim.Adam(text_encoder.parameters(),
                                        lr=args.learning_rate,
                                        betas=(0.5, 0.999))
        elif args.model == 'AD_LLM':
            trained_parameters = []
            for p in model.parameters():
                if p.requires_grad is True:
                    trained_parameters.append(p)
            total_params = sum(p.numel() for p in trained_parameters)
            print(f'Total parameters: {total_params}')#41035635

            # 相当于原代码中的生成器的优化器
            model_optim = optim.Adam(trained_parameters, lr=args.learning_rate, betas=(0.5, 0.999))
        else:
            gen_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        ##########
        # 判别器的优化器
        dis_optimizer = torch.optim.Adam(discriminator.parameters(),
                                         lr=args.learning_rate * 0.1,
                                         betas=(0.5, 0.999))
        #########

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        word_embeddings = llm_model.get_input_embeddings().weight
        vocab_size = word_embeddings.shape[0]
        embed_space_evaluator = EmbeddingSpaceEvaluator(args, eval_net_path, word_embeddings, vocab_size, device)
        # embed_space_evaluator = EmbeddingSpaceEvaluator(args_eval, eval_net_path, lang_model, device)

        train_loader, test_loader,  model, model_optim, scheduler = accelerator.prepare(
            train_loader, test_loader, model, model_optim, scheduler)

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # bc = AverageMeter('bc')
        loss_meters = [AverageMeter('loss'), AverageMeter('var_loss'), AverageMeter('gen'), AverageMeter('dis'),
                       AverageMeter('KLD'), AverageMeter('DIV_REG'), AverageMeter('c_pos'),
                       AverageMeter('c_neg'), AverageMeter('phy')]
        iter_count = 0
        for epoch in range(args.train_epochs):
            # val_loss, val_mae_val, val_frechet_dist, BC, diversity_score = evaluate_testset(test_loader, model, embed_space_evaluator, epoch, speaker_model, args)
            train_loss = []
            dis_loss = []
            # model.train()
            epoch_time = time.time()

            for i, data in tqdm(enumerate(train_loader, 0)):
                iter_count += 1
                # model_optim.zero_grad()

                in_text, text_lengths, in_text_padded, text_token_padded, text, _, target_vec, in_audio, log_melspec, in_spec, aux_info = data
                batch_size = target_vec.size(0)
                in_text_padded = in_text_padded.to(accelerator.device)
                text_token_padded = text_token_padded.to(accelerator.device)
                in_audio = in_audio.to(accelerator.device)
                target_dir_vec = target_vec.to(accelerator.device)
                log_melspec = log_melspec.to(accelerator.device)
                pre_seq = target_dir_vec[:, 0:16]

                ######
                # vid_indices = []
                if speaker_model:
                    vids = aux_info['vid']
                    vid_indices = [speaker_model.word2index[vid] for vid in vids]
                    vid_indices = torch.LongTensor(vid_indices).to(device)
                ######
                if args.generator == 'LLM_generator':
                    loss = train_llm(args, epoch, in_audio, log_melspec, text_token_padded, target_dir_vec, vid_indices,
                                      model, discriminator,
                                      model_optim, dis_optimizer, accelerator)
                if args.generator == 'multimodal_context':
                    # 这里面generator就相当于我们做的神经网络
                    loss = train_iter_gan(args, epoch, in_text_padded, in_audio, target_dir_vec, vid_indices,
                                          generator, discriminator,
                                          gen_optimizer, dis_optimizer)
                elif args.model == 'joint_embedding':
                    loss = train_iter_embed(args, epoch, in_text_padded, in_audio, target_dir_vec,
                                            generator, gen_optimizer, mode='random')
                elif args.model == 'gesture_autoencoder':
                    loss = train_iter_embed(args, epoch, in_text_padded, in_audio, target_dir_vec,
                                            generator, gen_optimizer)
                elif args.model == 'seq2seq':
                    loss = train_iter_seq2seq(args, epoch, in_text, text_lengths, target_dir_vec, generator, gen_optimizer)
                elif args.model == 'speech2gesture':
                    loss = train_iter_speech2gesture(args, in_spec, target_dir_vec, generator, discriminator,
                                                     gen_optimizer, dis_optimizer, loss_fn)

                # loss values
                for loss_meter in loss_meters:
                    name = loss_meter.name
                    if name in loss:
                        loss_meter.update(loss[name], batch_size)
                # print('lossssss',loss['loss'])

                if (i + 1) % 100 == 0:
                    print_summary = "\titers: {0}, epoch: {1} ".format(i + 1, epoch + 1)
                    for loss_meter in loss_meters:
                        if loss_meter.count > 0:
                            print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                            loss_meter.reset()
                    print(print_summary)
                    # accelerator.print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    # iter_count = 0
                    time_now = time.time()

            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f}".format(
                    epoch + 1, train_loss))

            val_loss, val_mae_val, val_frechet_dist, BC, diversity_score = evaluate_testset(test_loader, model, embed_space_evaluator, epoch, speaker_model, args)

            tb_writer.add_scalar('diversity_score/val', diversity_score, epoch)
            tb_writer.add_scalar('val_frechet_dist/val', val_frechet_dist, epoch)
            tb_writer.add_scalar('BC/val', BC, epoch)

            eval_dict['frechet'] = val_frechet_dist
            if eval_dict['frechet'] < best_values['frechet']:#4 and BC > 0.7:#best_values['frechet']:
                best_values['frechet'] = eval_dict['frechet']

                gen_state_dict = model.state_dict()
                filename = '/home/wxp/chy/HOP/save_checkpoint/expressive_FGD_{:.2f}.bin'.format(eval_dict['frechet'])
                # filename = '/home/wxp/chy/HOP/save_checkpoint/the_best.bin'
                torch.save({'generator': gen_state_dict}, filename)
                print('Saved the checkpoint')
                logging.info('Saved the checkpoint')
            # if eval_dict['frechet'] < best_values['frechet']:
            #     best_values['frechet'] = eval_dict['frechet']
            print('  *** BEST VALIDATION FGD: {:.3f}'.format(best_values['frechet']))

        tb_writer.close()

# accelerator.wait_for_everyone()
# if accelerator.is_local_main_process:
#     path = './checkpoints'  # unique checkpoint saving path
#     del_files(path)  # delete checkpoint files
#     accelerator.print('success delete che