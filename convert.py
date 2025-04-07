import os
import time
import torch
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from scipy.interpolate import interp1d
import subprocess
from textwrap import wrap
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dir_vec_pairs = [(0, 1, 0.26), (1, 2, 0.18), (2, 3, 0.14), (1, 4, 0.22), (4, 5, 0.36),
                 (5, 6, 0.33), (1, 7, 0.22), (7, 8, 0.36), (8, 9, 0.33)]  # adjacency and bone length

def get_words_in_time_range(word_list, start_time, end_time):
    words = []

    for word in word_list:
        _, word_s, word_e = word[0], word[1], word[2]

        if word_s >= end_time:
            break

        if word_e <= start_time:
            continue

        words.append(word)

    return words

def convert_pose_seq_to_dir_vec(pose):
    if pose.shape[-1] != 3:
        pose = pose.reshape(pose.shape[:-1] + (-1, 3))

    if len(pose.shape) == 3:
        dir_vec = np.zeros((pose.shape[0], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, i] = pose[:, pair[1]] - pose[:, pair[0]]
            dir_vec[:, i, :] = normalize(dir_vec[:, i, :], axis=1)  # to unit length
    elif len(pose.shape) == 4:  # (batch, seq, ...)
        dir_vec = np.zeros((pose.shape[0], pose.shape[1], len(dir_vec_pairs), 3))
        for i, pair in enumerate(dir_vec_pairs):
            dir_vec[:, :, i] = pose[:, :, pair[1]] - pose[:, :, pair[0]]
        for j in range(dir_vec.shape[0]):  # batch
            for i in range(len(dir_vec_pairs)):
                dir_vec[j, :, i, :] = normalize(dir_vec[j, :, i, :], axis=1)  # to unit length
    else:
        assert False

    return dir_vec

# 得到随机种子必须的函数

def resample_pose_seq(poses, duration_in_sec, fps):
    n = len(poses)
    x = np.arange(0, n)
    y = poses
    f = interp1d(x, y, axis=0, kind='linear', fill_value='extrapolate')
    expected_n = duration_in_sec * fps
    x_new = np.arange(0, n, n / expected_n)
    interpolated_y = f(x_new)
    if hasattr(poses, 'dtype'):
        interpolated_y = interpolated_y.astype(poses.dtype)
    return interpolated_y

def convert_dir_vec_to_pose(vec):
    # vec = np.array(vec)
    #
    # if vec.shape[-1] != 3:
    #     vec = vec.reshape(vec.shape[:-1] + (-1, 3))
    #
    # if len(vec.shape) == 2:
    #     joint_pos = np.zeros((10, 3))
    #     for j, pair in enumerate(dir_vec_pairs):
    #         joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]
    # elif len(vec.shape) == 3:
    #     joint_pos = np.zeros((vec.shape[0], 10, 3))
    #     for j, pair in enumerate(dir_vec_pairs):
    #         joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
    # elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
    #     joint_pos = np.zeros((vec.shape[0], vec.shape[1], 10, 3))
    #     for j, pair in enumerate(dir_vec_pairs):
    #         joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
    # else:
    #     assert False

    #######
    if vec.shape[-1] != 3:
        vec = vec.reshape(vec.shape[:-1] + (-1, 3))

    if len(vec.shape) == 2:
        joint_pos = torch.zeros((10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[pair[1]] = joint_pos[pair[0]] + pair[2] * vec[j]
    elif len(vec.shape) == 3:
        joint_pos = torch.zeros((vec.shape[0], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, pair[1]] = joint_pos[:, pair[0]] + pair[2] * vec[:, j]
    elif len(vec.shape) == 4:  # (batch, seq, 9, 3)
        joint_pos = torch.zeros((vec.shape[0], vec.shape[1], 10, 3))
        for j, pair in enumerate(dir_vec_pairs):
            joint_pos[:, :, pair[1]] = joint_pos[:, :, pair[0]] + pair[2] * vec[:, :, j]
    else:
        assert False
    #######

    return joint_pos


def create_video_and_save(save_path, epoch, prefix, iter_idx, target, output, mean_data, title,
                          audio=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=True):
    print('rendering a video...')
    start = time.time()

    fig = plt.figure(figsize=(8, 4))
    axes = [fig.add_subplot(1, 2, 1, projection='3d'), fig.add_subplot(1, 2, 2, projection='3d')]
    axes[0].view_init(elev=20, azim=-60)
    axes[1].view_init(elev=20, azim=-60)
    fig_title = title

    if aux_str:
        fig_title += ('\n' + aux_str)
    fig.suptitle('\n'.join(wrap(fig_title, 75)), fontsize='medium')

    # un-normalization and convert to poses
    mean_data = mean_data.flatten()
    output = output + mean_data
    output_poses = convert_dir_vec_to_pose(output)
    target_poses = None
    if target is not None:
        target = target + mean_data
        target_poses = convert_dir_vec_to_pose(target)

    def animate(i):
        for k, name in enumerate(['human', 'generated']):
            if name == 'human' and target is not None and i < len(target):
                pose = target_poses[i]
            elif name == 'generated' and i < len(output):
                pose = output_poses[i]
            else:
                pose = None

            if pose is not None:
                axes[k].clear()
                # axes[k].axis('off')  # 隐藏坐标轴
                for j, pair in enumerate(dir_vec_pairs):
                    axes[k].plot([pose[pair[0], 0], pose[pair[1], 0]],
                                 [pose[pair[0], 2], pose[pair[1], 2]],
                                 [pose[pair[0], 1], pose[pair[1], 1]],
                                 zdir='z', linewidth=5)
                axes[k].set_xlim3d(-0.5, 0.5)
                axes[k].set_ylim3d(0.5, -0.5)
                axes[k].set_zlim3d(0.5, -0.5)

                # axes[k].set_xticks([])  # 设置 x 轴的刻度：从 -0.5 到 0.5，步长为 0.1
                # axes[k].set_yticks([])
                # axes[k].set_zticks([])

                # # 设置不显示刻度标签
                # axes[k].set_xticklabels([])
                # axes[k].set_yticklabels([])
                # axes[k].set_zticklabels([])

                axes[k].set_xlabel('')
                axes[k].set_ylabel('')
                axes[k].set_zlabel('')

                axes[k].set_title('{} ({}/{})'.format(name, i + 1, len(output)))

    if target is not None:
        num_frames = max(len(target), len(output))
    else:
        num_frames = len(output)
    ani = animation.FuncAnimation(fig, animate, interval=30, frames=num_frames, repeat=False)

    # show audio
    audio_path = None
    if audio is not None:
        assert len(audio.shape) == 1  # 1-channel, raw signal
        audio = audio.astype(np.float32)
        sr = 16000
        audio_path = '{}/{}_{:03d}_{}.wav'.format(save_path, prefix, epoch, iter_idx)
        sf.write(audio_path, audio, sr)

        y, sr = librosa.load(audio_path, sr=None)
        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr)
        plt.savefig('{}/{}_{:03d}_{}.jpg'.format(save_path, prefix, epoch, iter_idx))

        # save video
    try:
        video_path = '{}/temp_{}_{:03d}_{}.mp4'.format(save_path, prefix, epoch, iter_idx)
        ani.save(video_path, fps=15, dpi=80)  # dpi 150 for a higher resolution
        del ani
        plt.close(fig)
    except RuntimeError:
        assert False, 'RuntimeError'

    # merge audio and video
    if audio is not None:
        merged_video_path = '{}/{}_{:03d}_{}.mp4'.format(save_path, prefix, epoch, iter_idx)
        cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', video_path, '-i', audio_path, '-strict', '-2',
               merged_video_path]
        if clipping_to_shortest_stream:
            cmd.insert(len(cmd) - 1, '-shortest')
        subprocess.call(cmd)
        if delete_audio_file:
            os.remove(audio_path)
        os.remove(video_path)

    print('done, took {:.1f} seconds'.format(time.time() - start))
    return output_poses, target_poses
