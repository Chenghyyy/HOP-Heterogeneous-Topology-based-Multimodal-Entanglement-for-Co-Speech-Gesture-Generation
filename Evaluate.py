import torch
import math
import time
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
import numpy as np
from utils.average_meter import AverageMeter
from tqdm import tqdm
import soundfile as sf
import librosa
import random
import torch.nn.functional as F
from convert import resample_pose_seq, convert_pose_seq_to_dir_vec, create_video_and_save, convert_dir_vec_to_pose, get_words_in_time_range

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
                 (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length

thres = 0.03
sigma = 0.1

def convert_dir_vec_to_pose(vec):
    vec = np.array(vec)

    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = np.zeros((10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = np.zeros((vec.shape[0], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
        joint_pos = np.zeros((vec.shape[0], vec.shape[1], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
    else:
        assert False

    return joint_pos

def evaluate_testset(test_data_loader, model, embed_space_evaluator, epoch, speaker_model, args):
    if args.datasets == 'TED':
        mean_dir_vec = [0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367,
                        -0.4121039, -0.9236511, 0.3061306, -0.0012415, -0.5155854, 0.8129665, 0.0871897, 0.2348464,
                        0.1846561, 0.8091402, 0.9271948, 0.2960011, -0.013189, 0.5233978, 0.8092403, 0.0725451,
                        -0.2037076, 0.1924306, 0.8196916]

        angle_pair = [
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8)
        ]
        change_angle = [0.0034540758933871984, 0.007043459918349981, 0.003493624273687601, 0.007205077446997166]
    else:
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
                        0.022748252376914024, 0.039307352155447006, 0.03733552247285843, 0.03775784373283386,
                        0.0485558956861496,
                        0.032914578914642334, 0.03800227493047714, 0.03757007420063019, 0.027338404208421707,
                        0.01640886254608631,
                        0.003166505601257086, 0.0017252820543944836, 0.0018696568440645933, 0.0016072227153927088,
                        0.005681346170604229,
                        0.013287615962326527, 0.021516695618629456, 0.033936675637960434, 0.03094293735921383,
                        0.03378918394446373,
                        0.044323261827230453, 0.034706637263298035, 0.03369896858930588, 0.03573163226246834,
                        0.02628341130912304,
                        0.014071882702410221, 0.0029828345868736506, 0.0015706412959843874, 0.0017107439925894141,
                        0.0014634154504165053,
                        0.004873405676335096, 0.002998138777911663, 0.0030240598134696484, 0.0009890805231407285,
                        0.0012279648799449205,
                        0.047324635088443756, 0.04472292214632034]

        # 得到随机种子必须的向量
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

    model.train(False)
    loss_val = []
    mae = []
    bc = AverageMeter('bc')

    embed_space_evaluator.reset()
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            in_text, text_lengths, in_text_padded, text_token_padded, text, _, target_vec, in_audio, log_melspec, in_spec, aux_info = data
            batch_size = target_vec.size(0)
            in_text_padded = in_text_padded.to(device)
            text_token_padded = text_token_padded.to(device)
            in_audio = in_audio.to(device)

            target_dir_vec = target_vec.to(device)
            log_melspec = log_melspec.to(device)

            pre_seq = target_dir_vec[:, 0:16]
            # pre_seq[:, 0:4, -1] = 1  # indicating bit for constraints

            if speaker_model:
                vid_indices = [random.choice(list(speaker_model.word2index.values())) for _ in range(batch_size)]
                vid_indices = torch.LongTensor(vid_indices).to(device)
            else:
                vid_indices = None

            outputs, _, _, _ = model(in_audio, log_melspec, in_text_padded, pre_seq, vid_indices)

            if epoch > 35:
                if args.datasets == 'TED':
                    beat_vec = outputs.cpu().numpy() + np.array(mean_dir_vec).squeeze()
                    beat_vec = beat_vec.reshape(beat_vec.shape[0], beat_vec.shape[1], -1, 3)
                    beat_vec = torch.tensor(beat_vec).to(device)
                    beat_vec = F.normalize(beat_vec, dim=-1)
                    all_vec = beat_vec.reshape(beat_vec.shape[0] * beat_vec.shape[1], -1, 3)

                    for idx, pair in enumerate(angle_pair):
                        vec1 = all_vec[:, pair[0]]
                        vec2 = all_vec[:, pair[1]]
                        inner_product = torch.einsum('ij,ij->i', [vec1, vec2])
                        inner_product = torch.clamp(inner_product, -1, 1, out=None)
                        angle = torch.acos(inner_product) / math.pi
                        angle_time = angle.reshape(batch_size, -1)
                        if idx == 0:
                            angle_diff = torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(
                                change_angle)
                        else:
                            angle_diff += torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(
                                change_angle)
                    angle_diff = torch.cat((torch.zeros(batch_size, 1).to(device), angle_diff), dim=-1)

                    for b in range(batch_size):
                        motion_beat_time = []
                        for t in range(2, 33):
                            if (angle_diff[b][t] < angle_diff[b][t - 1] and angle_diff[b][t] < angle_diff[b][t + 1]):
                                if (angle_diff[b][t - 1] - angle_diff[b][t] >= thres or angle_diff[b][t + 1] - angle_diff[b][t] >= thres):
                                    motion_beat_time.append(float(t) / 15.0)
                        if (len(motion_beat_time) == 0):
                            continue
                        audio = in_audio[b].cpu().numpy()
                        audio_beat_time = librosa.onset.onset_detect(y=audio, sr=16000, units='time')

                        sum = 0
                        for audio in audio_beat_time:
                            # print('audio',audio)
                            # print('motion_beat_time',motion_beat_time)
                            sum += np.power(math.e, -np.min(np.power((audio - motion_beat_time), 2)) / (2 * sigma * sigma))
                        bc.update(sum / len(audio_beat_time), len(audio_beat_time))
                else:
                    beat_vec = outputs.cpu().numpy() + np.array(mean_dir_vec).squeeze()
                    beat_vec = torch.from_numpy(beat_vec).to(device)
                    left_palm = torch.cross(beat_vec[:, :, 11 * 3: 12 * 3], beat_vec[:, :, 17 * 3: 18 * 3], dim=2)
                    right_palm = torch.cross(beat_vec[:, :, 28 * 3: 29 * 3], beat_vec[:, :, 34 * 3: 35 * 3], dim=2)
                    beat_vec = torch.cat((beat_vec, left_palm, right_palm), dim=2)
                    beat_vec = beat_vec.reshape(beat_vec.shape[0], beat_vec.shape[1], -1, 3)
                    beat_vec = F.normalize(beat_vec, dim=-1)
                    all_vec = beat_vec.reshape(beat_vec.shape[0] * beat_vec.shape[1], -1, 3)

                    for idx, pair in enumerate(angle_pair):
                        vec1 = all_vec[:, pair[0]]
                        vec2 = all_vec[:, pair[1]]
                        inner_product = torch.einsum('ij,ij->i', [vec1, vec2])
                        inner_product = torch.clamp(inner_product, -1, 1, out=None)
                        angle = torch.acos(inner_product) / math.pi
                        angle_time = angle.reshape(batch_size, -1)
                        if idx == 0:
                            angle_diff = torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(
                                change_angle)
                        else:
                            angle_diff += torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(
                                change_angle)
                    angle_diff = torch.cat((torch.zeros(batch_size, 1).to(device), angle_diff), dim=-1)

                    for b in range(batch_size):
                        motion_beat_time = []
                        for t in range(2, 33):
                            if (angle_diff[b][t] < angle_diff[b][t - 1] and angle_diff[b][t] < angle_diff[b][t + 1]):
                                if (angle_diff[b][t - 1] - angle_diff[b][t] >= thres or angle_diff[b][t + 1] -
                                        angle_diff[b][t] >= thres):
                                    motion_beat_time.append(float(t) / 15.0)
                        if (len(motion_beat_time) == 0):
                            continue
                        audio = in_audio[b].cpu().numpy()
                        audio_beat_time = librosa.onset.onset_detect(y=audio, sr=16000, units='time')
                        sum = 0
                        for audio in audio_beat_time:
                            sum += np.power(math.e,
                                            -np.min(np.power((audio - motion_beat_time), 2)) / (2 * sigma * sigma))
                        bc.update(sum / len(audio_beat_time), len(audio_beat_time))

            loss = F.l1_loss(outputs, target_dir_vec)

            # 评估
            embed_space_evaluator.push_samples(text_token_padded, in_audio, outputs, target_dir_vec)
            # calculate MAE of joint coordinates
            out_dir_vec = outputs.cpu().numpy()
            out_dir_vec += np.array(mean_dir_vec).squeeze()
            out_joint_poses = convert_dir_vec_to_pose(out_dir_vec)

            target_vec_pose = target_vec.cpu().numpy()
            target_vec_pose += np.array(mean_dir_vec).squeeze()
            target_poses = convert_dir_vec_to_pose(target_vec_pose)

            if out_joint_poses.shape[1] == 34:
                diff = out_joint_poses[:, 4:] - target_poses[:, 4:]
            else:
                diff = out_joint_poses - target_poses[:, 4:]
            mae_val = np.mean(np.absolute(diff))

            loss_val.append(loss.item())
            mae.append(mae_val.item())


    loss = np.average(loss_val)
    mae_val = np.average(mae)
    frechet_dist, feat_dist = embed_space_evaluator.get_scores()
    diversity_score = embed_space_evaluator.get_diversity_scores()

    elapsed_time = time.time() - start
    print('[VAL] loss: {:.5f}, joint mae: {:.5f}, FGD: {:.5f}, feat_D: {:.5f}, BC: {:.4f} / {:.1f}s, Diversity: {:.3f}'.format(
                loss, mae_val, frechet_dist, feat_dist, bc.avg, elapsed_time, diversity_score))

    BC = bc.avg
    model.train(True)
    return loss, mae_val, frechet_dist, BC, diversity_score