import pickle
import argparse
import torch
import soundfile as sf
import lmdb
import pyarrow
import pandas as pd
import math
import librosa.display
from sklearn.preprocessing import StandardScaler

import utils.data_utils_expressive
from utils import train_utils_expressive
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from tqdm import tqdm

from model import HOP
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from utils.tools import del_files
from Evaluate import evaluate_testset

from data_loader.lmdb_data_loader_expressive import *
# from data_loader.lmdb_data_loader import *

import time
import random
import numpy as np
import os

from model.EmbeddingSpaceEvaluator import EmbeddingSpaceEvaluator
from load_checkpoint import load_checkpoint_and_model
from model.multimodal_context_net import ConvDiscriminator
from convert import resample_pose_seq, convert_pose_seq_to_dir_vec, create_video_and_save, convert_dir_vec_to_pose, get_words_in_time_range

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

parser = argparse.ArgumentParser(description='Time-LLM')

vid_idx = random.sample(range(0, 1000), 1)[0]

#为了确保程序中的随机性操作在每次运行时产生相同的结果，即“固定随机种子”。这样做可以使结果可复现，特别是在调试和实验过程中
# fix_seed = 2021
# random.seed(fix_seed)
# torch.manual_seed(fix_seed)
# np.random.seed(fix_seed)

'''
--task_name
long_term_forecast
--is_training
1
--root_path
./dataset/rhythm_dataset
--data_path
rhythm.csv
--model_id
ETTh1_512_96
--model
TimeLLM
--data
ETTh1
--features
M
--label_len
48
--seq_len
32
--pred_len
6
--factor
3
--enc_in
7
--dec_in
7
--c_out
7
--des
'Exp'
--itr
1
--d_model
128
--d_ff
128
--learning_rate
0.005
--llm_layers
12
--train_epochs
75
--model_comment
'TimeLLM-ETTh1'
'''

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
parser.add_argument('--datasets_path', type=str, default='/home/wxp/data/ted_expressive_dataset/train',
                    help='options: [./data/ted_dataset/lmdb_train,/home/wxp/data/ted_expressive_dataset/train]')
parser.add_argument('--datasets', type=str, default='TED_expressive',help='options: [TED, TED_expressive]')
parser.add_argument('--n_poses', type=int, default=34)
parser.add_argument('--pose_dim', type=int, default=126, help='TED-27,TED_expressive-126')
parser.add_argument('--wordembed_dim', type=int, default=300)
parser.add_argument('--n_pre_poses', type=int, default=4)
parser.add_argument("--z_type", type=str, default='speaker')

parser.add_argument("--loss_regression_weight", type=float, default=600, help='TED-600,TED_expressive-')
parser.add_argument("--loss_gan_weight", type=float, default=5)
parser.add_argument("--loss_kld_weight", type=float, default=0.6)
parser.add_argument("--loss_reg_weight", type=float, default=0.4)
parser.add_argument('--generator',type=str, default='LLM_generator',
                    help='model name, options: [hierarchy, multimodal_context, joint_embedding,gesture_autoencoder,seq2seq, speech2gesture]')
parser.add_argument("--use_gwnet", default=True)
parser.add_argument('--use_reprograme', default=True)

# optimization
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=75, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
                 (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length
if args.datasets == 'TED':
    mean_dir_vec = [ 0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039, -0.9236511, 0.3061306, -0.0012415, -0.5155854,  0.8129665,  0.0871897, 0.2348464,  0.1846561,  0.8091402,  0.9271948,  0.2960011, -0.013189 ,  0.5233978,  0.8092403,  0.0725451, -0.2037076, 0.1924306,  0.8196916]
    mean_pose = [0.0000306, 0.0004946, 0.0008437, 0.0033759, -0.2051629, -0.0143453, 0.0031566, -0.3054764, 0.0411491,
                 0.0029072, -0.4254303, -0.001311, -0.1458413, -0.1505532, -0.0138192, -0.2835603, 0.0670333, 0.0107002,
                 -0.2280813, 0.112117, 0.2087789, 0.1523502, -0.1521499, -0.0161503, 0.291909, 0.0644232, 0.0040145,
                 0.2452035, 0.1115339, 0.2051307]
    data_mean_dir_vec = np.array(mean_dir_vec).squeeze()
    val_mean_dir_vec = np.array(mean_dir_vec).reshape(-1, 3)
    eval_net_path = '/home/wxp/chy/text_evaluator/output/train_h36m_gesture_autoencoder/gesture_autoencoder_checkpoint_best.bin'
else:
    mean_dir_vec = [-0.0737964, -0.9968923, -0.1082858, 0.9111595, 0.2399522, -0.102547, -0.8936886, 0.3131501,
                    -0.1039348, 0.2093927, 0.958293, 0.0824881, -0.1689021, -0.0353824, -0.7588258, -0.2794763,
                    -0.2495191, -0.614666, -0.3877234, 0.005006, -0.5301695, -0.5098616, 0.2257808, 0.0053111,
                    -0.2393621, -0.1022204, -0.6583039, -0.4992898, 0.1228059, -0.3292085, -0.4753748, 0.2132857,
                    0.1742853, -0.2062069, 0.2305175, -0.5897119, -0.5452555, 0.1303197, -0.2181693, -0.5221036,
                    0.1211322, 0.1337591, -0.2164441, 0.0743345, -0.6464546, -0.5284583, 0.0457585, -0.319634,
                    -0.5074904, 0.1537192, 0.1365934, -0.4354402, -0.3836682, -0.3850554, -0.4927187, -0.2417618,
                    -0.3054556, -0.3556116, -0.281753, -0.5164358, -0.3064435, 0.9284261, -0.067134, 0.2764367,
                    0.006997, -0.7365526, 0.2421269, -0.225798, -0.6387642, 0.3788997, 0.0283412, -0.5451686,
                    0.5753376, 0.1935219, 0.0632555, 0.2122412, -0.0624179, -0.6755542, 0.5212831, 0.1043523,
                    -0.345288, 0.5443628, 0.128029, 0.2073687, 0.2197118, 0.2821399, -0.580695, 0.573988, 0.0786667,
                    -0.2133071, 0.5532452, -0.0006157, 0.1598754, 0.2093099, 0.124119, -0.6504359, 0.5465003,
                    0.0114155, -0.3203954, 0.5512083, 0.0489287, 0.1676814, 0.4190787, -0.4018607, -0.3912126,
                    0.4841548, -0.2668508, -0.3557675, 0.3416916, -0.2419564, -0.5509825, 0.0485515, -0.6343101,
                    -0.6817347, -0.4705639, -0.6380668, 0.4641643, 0.4540192, -0.6486361, 0.4604001, -0.3256226,
                    0.1883097, 0.8057457, 0.3257385, 0.1292366, 0.815372]
    mean_pose = [-0.0046788, -0.5397806, 0.007695, -0.0171913, -0.7060388, -0.0107034, 0.1550734, -0.6823077,
                 -0.0303645, -0.1514748, -0.6819547, -0.0268262, 0.2094328, -0.469447, -0.0096073, -0.2318253,
                 -0.4680838, -0.0444074, 0.1667382, -0.4643363, -0.1895118, -0.1648597, -0.4552845, -0.2159728,
                 0.1387546, -0.4859474, -0.2506667, 0.1263615, -0.4856088, -0.2675801, 0.1149031, -0.4804542, -0.267329,
                 0.1414847, -0.4727709, -0.2583424, 0.1262482, -0.4686185, -0.2682536, 0.1150217, -0.4633611,
                 -0.2640182, 0.1475897, -0.4415648, -0.2438853, 0.1367996, -0.4383164, -0.248248, 0.1267222, -0.435534,
                 -0.2455436, 0.1455485, -0.4557491, -0.2521977, 0.1305471, -0.4535603, -0.2611591, 0.1184687,
                 -0.4495366, -0.257798, 0.1451682, -0.4802511, -0.2081622, 0.1301337, -0.4865308, -0.2175783, 0.1208341,
                 -0.4932623, -0.2311025, -0.1409241, -0.4742868, -0.2795303, -0.1287992, -0.4724431, -0.2963172,
                 -0.1159225, -0.4676439, -0.2948754, -0.1427748, -0.4589126, -0.2861245, -0.126862, -0.4547355,
                 -0.2962466, -0.1140265, -0.451308, -0.2913815, -0.1447202, -0.4260471, -0.2697673, -0.1333492,
                 -0.4239912, -0.2738043, -0.1226859, -0.4238346, -0.2706725, -0.1446909, -0.440342, -0.2789209,
                 -0.1291436, -0.4391063, -0.2876539, -0.1160435, -0.4376317, -0.2836147, -0.1441438, -0.4729031,
                 -0.2355619, -0.1293268, -0.4793807, -0.2468831, -0.1204146, -0.4847246, -0.2613876, -0.0056085,
                 -0.9224338, -0.1677302, -0.0352157, -0.963936, -0.1388849, 0.0236298, -0.9650772, -0.1385154,
                 -0.0697098, -0.9514691, -0.055632, 0.0568838, -0.9565502, -0.0567985]

    # mean_dir_vec = np.array(mean_dir_vec).reshape(-1, 3)
    data_mean_dir_vec = np.array(mean_dir_vec).squeeze()
    val_mean_dir_vec = np.array(mean_dir_vec).reshape(-1, 3)
    eval_net_path = '/home/wxp/chy/text_evaluator/output/train_h36m_gesture_autoencoder/ted_expressive_autoencoder_checkpoint_best.bin'

checkpoint_path = '/home/wxp/chy/Gesture-Generation-from-Trimodal-Context-master/output/train_multimodal_context/multimodal_context_checkpoint_best.bin'
our_checkpoint_path = '/home/wxp/chy/text_timellm/save_checkpoint/expressive_FGD_1.83.bin'

save_video_path = '/home/wxp/chy/text_timellm/test_video'

our_checkpoint = torch.load(our_checkpoint_path, map_location=device)

if args.llm_model == 'LLAMA':
    # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
    llama_config = LlamaConfig.from_pretrained("/home/wxp/chy/text_timellm/llama_model/")
    # self.llama_config = LlamaConfig.from_pretrained("/home/wxp/LLaMA-Factory/merge_models/llama_lora_sft/")
    llama_config.num_hidden_layers = args.llm_layers
    llama_config.output_attentions = True
    llama_config.output_hidden_states = True
    try:
        llm_model = LlamaModel.from_pretrained(
            # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
            "/home/wxp/chy/text_timellm/llama_model/",
            # "/home/wxp/LLaMA-Factory/merge_models/llama_lora_sft/",
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
            "/home/wxp/chy/text_timellm/llama_model/",
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

args_eval, _, lang_model, _, _ = load_checkpoint_and_model(checkpoint_path, device)
out_dim = args.pose_dim
discriminator = ConvDiscriminator(out_dim).to(device)

train_dataset = SpeechMotionDataset(args.datasets_path,#'./data/ted_dataset/lmdb_train'#'/home/wxp/data/ted_expressive_dataset/train'
                                    n_poses=34,
                                    subdivision_stride=10,
                                    pose_resampling_fps=15,
                                    mean_dir_vec=val_mean_dir_vec,
                                    mean_pose=mean_pose,
                                    tokenizer=tokenizer,
                                    remove_word_timing=('text')
                                    )

speaker_model = train_dataset.speaker_model

model = HOP.Model(args, llm_model, tokenizer, speaker_model).float()
model.to(device)  # 加gru后
model.load_state_dict(our_checkpoint['generator'])
model.train(False)

# save_path = '/home/wxp/chy/text_timellm/test_video'
if args.datasets == 'TED':
    test_data_path = '/home/wxp/chy/text_timellm/data/ted_dataset/lmdb_test'
else:
    test_data_path = '/home/wxp/data/ted_expressive_dataset/test'

n_saved = 0
n_generations = 1
lmdb_env = lmdb.open(test_data_path, readonly=True, lock=False)

while n_saved < n_generations:
    n_saved += 1
    with lmdb_env.begin(write=False) as txn:
        keys = [key for key, _ in txn.cursor()]
        # select video
        # key = random.choice(keys)
        # print('key',key)
        key = b'0000001333'
        buf = txn.get(key)
        video = pyarrow.deserialize(buf)
        vid = video['vid']
        clips = video['clips']
    n_clips = len(clips)
    print('n_clips', n_clips)
    if n_clips == 0:
        continue
    # clip_idx = random.randrange(n_clips)
    # print('clip_idx',clip_idx)
    clip_idx = 0
    clip_audio = clips[clip_idx]['audio_raw']
    clip_words = clips[clip_idx]['words']
    clip_time = [clips[clip_idx]['start_time'], clips[clip_idx]['end_time']]
    clip_poses = clips[clip_idx]['skeletons_3d']

    # synthesize
    for selected_vi in range(len(clip_words)):  # make start time of input text zero
        clip_words[selected_vi][1] -= clip_time[0]  # start time
        clip_words[selected_vi][2] -= clip_time[0]  # end time

    if args.datasets == 'TED':
        print('TED')
        clip_poses = resample_pose_seq(clip_poses, clip_time[1] - clip_time[0], 15)
        target_dir_vec = convert_pose_seq_to_dir_vec(clip_poses)
    else:
        print('TED_expressive')
        clip_poses = utils.data_utils_expressive.resample_pose_seq(clip_poses, clip_time[1] - clip_time[0], 15)
        target_dir_vec = utils.data_utils_expressive.convert_pose_seq_to_dir_vec(torch.from_numpy(clip_poses))

    target_dir_vec = target_dir_vec.reshape(target_dir_vec.shape[0], -1)
    target_dir_vec -= data_mean_dir_vec
    target_dir_vec = torch.from_numpy(target_dir_vec)

    out_list = []
    n_frames = 34
    clip_length = len(clip_audio) / 16000
    pre_seq = torch.Tensor(target_dir_vec[0:16])
    pre_seq = pre_seq.unsqueeze(0)

    sr = 16000
    # divide into synthesize units and do synthesize
    unit_time = 34 / 15
    stride_time = (34 - 4) / 15
    if clip_length < unit_time:
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / stride_time) + 1
    spectrogram_sample_length = int(round(unit_time * sr / 512))
    audio_sample_length = int(unit_time * 16000)
    end_padding_duration = 0

    vid = random.randrange(model.z_obj.n_words)
    # vid = random.randrange(speaker_model.n_words)
    print('vid:', vid)
    vid = torch.LongTensor([vid]).to(device)

    print('{}, {}, {}, {}, {}'.format(num_subdivision, unit_time, clip_length, stride_time, audio_sample_length))

    outputs = None
    for a in range(0, num_subdivision):  # num_subdivision应该代表的是将数据集划分为几块
        start_time = a * stride_time  # stride_time应该代表的是每块的时间
        end_time = start_time + unit_time

        # prepare audio input
        audio_start = math.floor(start_time / clip_length * len(clip_audio))
        audio_end = audio_start + audio_sample_length
        in_audio = clip_audio[audio_start:audio_end]
        #########
        #为了凑维度加的
        in_audio_test = np.pad(in_audio, (0, audio_sample_length-len(in_audio)), 'constant')
        #########
        assert len(in_audio.shape) == 1  # 1-channel, raw signal

        melspec = librosa.feature.melspectrogram(y=in_audio_test, sr=16000, n_fft=1024, hop_length=1096, power=2)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time
        log_melspec = log_melspec.T
        log_melspec = torch.from_numpy(log_melspec).to("cuda:0").float()
        log_melspec = log_melspec.unsqueeze(0)

        #####用于计算fgd
        in_audio = np.pad(in_audio, (0, audio_sample_length - len(in_audio)),'constant')  # 在in_audio数据末尾补0，补到长度为audio_sample_length - len(in_audio)，可能是因为音频数据自带时间信息
        in_audio = torch.from_numpy(in_audio).unsqueeze(0).to("cuda:0").float()
        #####

        # prepare text input
        word_seq = get_words_in_time_range(word_list=clip_words, start_time=start_time, end_time=end_time)
        text = []
        #####
        extended_word_indices = np.zeros(34)
        text_token_padded = np.zeros(34)
        frame_duration = (end_time - start_time) / 34
        #####
        i = 0
        for w_i, word in enumerate(word_seq):
            #####用于计算fgd
            idx = max(0, int(np.floor((word[1] - start_time) / frame_duration)))
            extended_word_indices[idx] = lang_model.get_word_index(word[0])
            #####
            text.append(word[0])
        text = ' '.join(text)
        print(text)

        text_token = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=2048, add_special_tokens=False).input_ids
        for w_i, word in enumerate(word_seq):
            idx = max(0, int(np.floor((word[1] - start_time) / frame_duration)))
            text_token_padded[idx] = text_token[0, w_i]
            i += 1
            if i == text_token.shape[1]:
                break
        in_text_padded = torch.LongTensor(extended_word_indices).unsqueeze(0).to("cuda:0")
        text_token_padded = torch.LongTensor(text_token_padded).unsqueeze(0).to("cuda:0")

        if a > 0:
            pre_seq = outputs[:, -16:]
            # pre_seq[0, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints
        pre_seq = pre_seq.float().to(device)

        time_now = time.time()
        iter_count = 0
        train_loss = []
        dis_loss = []
        epoch_time = time.time()

        outputs, _, _, _  = model(in_audio, log_melspec, text_token_padded, pre_seq, vid)
        out_seq = outputs[0, :, :].data.cpu().numpy()

        if len(out_list) > 0:
            last_poses = out_list[-1][-4:]
            out_list[-1] = out_list[-1][:-4]  # delete last 4 frames

            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[j]
                next = out_seq[j]
                out_seq[j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)

        out_list.append(out_seq)
    out_dir_vec = np.vstack(out_list)

    # make a video
    sentence_words = []
    for word, _, _ in clip_words:
        sentence_words.append(word)
    sentence = ' '.join(sentence_words)

    os.makedirs(save_video_path, exist_ok=True)

    # filename_prefix = '{}'.format(str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    filename_prefix = '{}_{}_{}'.format(args.datasets, key, clip_idx)

    filename_prefix_for_video = filename_prefix
    aux_str = '({})'.format(random.sample(range(0, 1000), 1)[0])

    # print('target_dir_vec',target_dir_vec.shape)
    target_dir_vec = target_dir_vec.to(torch.float32).cpu().numpy()
    # out_dir_vec = target_dir_vec
    if args.datasets == 'TED':
        print('TED123')
        create_video_and_save(
            save_video_path, 0, filename_prefix_for_video, 0, target_dir_vec, out_dir_vec,
            data_mean_dir_vec, sentence, audio=clip_audio, aux_str=aux_str,
            clipping_to_shortest_stream=True, delete_audio_file=False)
        # save pkl
        out_dir_vec = out_dir_vec + mean_dir_vec
        out_poses = convert_dir_vec_to_pose(out_dir_vec)
    else:
        print('TED_expressive')
        out_dir_vec = out_dir_vec + mean_dir_vec
        out_poses = utils.data_utils_expressive.convert_dir_vec_to_pose(out_dir_vec)


    save_dict = {
        'sentence': sentence, 'audio': clip_audio.astype(np.float32),
        'out_dir_vec': out_dir_vec, 'out_poses': out_poses,
        'aux_info': '{}_{}_{}'.format(vid, vid_idx, clip_idx),
        'human_dir_vec': target_dir_vec + mean_dir_vec,
    }
    with open(os.path.join(save_video_path, '{}.pkl'.format(filename_prefix)), 'wb') as f:
        pickle.dump(save_dict, f)

mean_pose = np.array(mean_pose).squeeze()

# '/home/wxp/data/ted_expressive_dataset/val'
#'./data/ted_dataset/lmdb_val'
if args.datasets == 'TED':
    print('TED')
    val_path = './data/ted_dataset/lmdb_val'
else:
    print('TED_expressive')
    val_path = '/home/wxp/data/ted_expressive_dataset/val'
val_dataset = SpeechMotionDataset(val_path,
                                  n_poses=34,
                                  subdivision_stride=10,
                                  pose_resampling_fps=15,
                                  speaker_model=speaker_model,
                                  mean_dir_vec=val_mean_dir_vec,
                                  mean_pose=mean_pose,
                                  tokenizer=tokenizer,
                                  )
test_loader = DataLoader(dataset=val_dataset, batch_size=256,
                         shuffle=False, drop_last=True, num_workers=4, pin_memory=True,
                         collate_fn=default_collate_fn
                         )

val_dataset.set_lang_model(lang_model)
epoch = 1000
word_embeddings = llm_model.get_input_embeddings().weight
vocab_size = word_embeddings.shape[0]
embed_space_evaluator = EmbeddingSpaceEvaluator(args, eval_net_path, word_embeddings, vocab_size, device)
val_loss, val_frechet_dist, val_mae_val, BC, diversity_score = evaluate_testset(test_loader, model, embed_space_evaluator, epoch, speaker_model, args)
