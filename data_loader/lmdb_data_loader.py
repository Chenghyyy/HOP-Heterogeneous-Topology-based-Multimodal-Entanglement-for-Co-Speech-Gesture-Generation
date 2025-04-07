import datetime
import logging
import os
import pickle
import random

from sklearn.preprocessing import StandardScaler
import soundfile as sf
import librosa
import pandas as pd
import numpy as np
import lmdb as lmdb
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

from data_loader import data_utils
from model.vocab import Vocab
from data_loader.data_preprocessor import DataPreprocessor
import pyarrow


def word_seq_collate_fn(data):
    """ collate function for loading word sequences in variable lengths """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    word_seq, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info = zip(*data)

    # merge sequences
    words_lengths = torch.LongTensor([len(x) for x in word_seq])
    word_seq = pad_sequence(word_seq, batch_first=True).long()

    text_padded = default_collate(text_padded)
    poses_seq = default_collate(poses_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return word_seq, words_lengths, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info


def default_collate_fn(data):
    _, text_padded, text_token_padded, text, pose_seq, vec_seq, audio_padded, log_melspec, spectrogram, aux_info = zip(*data)

    text_padded = default_collate(text_padded)
    text_token_padded = default_collate(text_token_padded)
    # audio = default_collate(audio)
    # text = ' '.join(text)
    pose_seq = default_collate(pose_seq)
    vec_seq = default_collate(vec_seq)
    audio_padded = default_collate(audio_padded)
    log_melspec = default_collate(log_melspec)

    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return torch.tensor([0]), torch.tensor([0]), text_padded, text_token_padded, text, pose_seq, vec_seq, audio_padded, log_melspec, spectrogram, aux_info


class SpeechMotionDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec, tokenizer,
                 speaker_model=None, remove_word_timing=False):

        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.mean_dir_vec = mean_dir_vec
        self.remove_word_timing = remove_word_timing
        self.tokenizer = tokenizer

        self.expected_audio_length = int(round(n_poses / pose_resampling_fps * 16000))
        self.expected_spectrogram_length = data_utils.calc_spectrogram_length_from_motion_length(
            n_poses, pose_resampling_fps)

        # self.lang_model = None
        self.lang_model = None

        logging.info("Reading data '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            logging.info('Creating the dataset cache...')
            assert mean_dir_vec is not None
            if mean_dir_vec.shape[-1] != 3:
                mean_dir_vec = mean_dir_vec.reshape(mean_dir_vec.shape[:-1] + (-1, 3))
            n_poses_extended = int(round(n_poses * 1.25))  # some margin
            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses_extended,
                                            subdivision_stride, pose_resampling_fps, mean_pose, mean_dir_vec)
            data_sampler.run()
        else:
            logging.info('Found the cache {}'.format(preloaded_dir))

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

        # make a speaker model
        if speaker_model is None or speaker_model == 0:
            precomputed_model = lmdb_dir + '_speaker_model.pkl'
            if not os.path.exists(precomputed_model):
                self._make_speaker_model(lmdb_dir, precomputed_model)
            else:
                with open(precomputed_model, 'rb') as f:
                    self.speaker_model = pickle.load(f)
        else:
            self.speaker_model = speaker_model

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            # print(sample)
            word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
            # print('key',key)
            # print('len_audio',len(audio))
            # print('audio',audio)

        def extend_word_seq(lang, words, end_time=None):
            n_frames = self.n_poses
            #####
            text = []
            #####
            if end_time is None:
                end_time = aux_info['end_time']
            frame_duration = (end_time - aux_info['start_time']) / n_frames

            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            text_token_padded = np.zeros(n_frames)#因为tokenizer会产生一个起始和结束字符，所以加个2
            #但是加2了会打乱原来的时间步长，音频的时间步长与文本的时间步长应该一样，也就是34
            #在tokenizer后加入参数, add_special_tokens=False可防止分词时加入起始索引

            if self.remove_word_timing:
                # print(123456)
                n_words = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    #####
                    text.append(word[0])
                    #####
                    if idx < n_frames:
                        n_words += 1

                text = ' '.join(text)
                text_token = self.tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=2048, add_special_tokens=False).input_ids
                # print(text)
                # print(text_token)
                # print(text_token.shape)#(1,x)

                # text_token_padded[0] = text_token[0, 0]
                # text_token_padded[33] = text_token[0,-1]

                # 查看0对应的标记
                # token_0 = self.tokenizer.convert_ids_to_tokens(0)
                # print(f"Token for ID 0: {token_0}")
                # # 查看特殊标记的索引
                # print(f"Padding token ID: {self.tokenizer.pad_token_id}")
                # print(f"Padding token: {self.tokenizer.pad_token}")
                # Token for ID 0: [PAD]
                # Padding token ID: 0
                # Padding token: [PAD]

                space = int(n_frames / (n_words + 1))
                for i in range(n_words):
                    idx = (i+1) * space
                    extended_word_indices[idx] = lang.get_word_index(words[i][0])
                    text_token_padded[idx] = text_token[0, i]
                # print('extended_word_indices', extended_word_indices)
                # print('text_padded',text_token_padded)

            else:
                # print('xyz')
                prev_idx = 0
                i = 0
                for word in words:
                    text.append(word[0])
                text = ' '.join(text)
                text_token = self.tokenizer(text, return_tensors="pt", padding=False, truncation=True,
                                            max_length=2048, add_special_tokens=False).input_ids
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        extended_word_indices[idx] = lang.get_word_index(word[0])
                        text_token_padded[idx] = text_token[0, i]
                        i += 1
                        # extended_word_indices[prev_idx:idx+1] = lang.get_word_index(word[0])
                        prev_idx = idx
                # print('extended_word_indices',extended_word_indices)
                # print('text_token_padded',text_token_padded)
            return torch.Tensor(extended_word_indices).long(), text_token_padded, text

        duration = aux_info['end_time'] - aux_info['start_time']
        do_clipping = True

        if do_clipping:
            sample_end_time = aux_info['start_time'] + duration * self.n_poses / vec_seq.shape[0]
            audio_padded = data_utils.make_audio_fixed_length(audio, self.expected_audio_length)#得到原代码中的in_audio_padded
            spectrogram = spectrogram[:, 0:self.expected_spectrogram_length]
            vec_seq = vec_seq[0:self.n_poses]
            pose_seq = pose_seq[0:self.n_poses]
        else:
            sample_end_time = None

        ##############
        # print(len(audio_padded))#44800/audio_padded_len=36267
        melspec = librosa.feature.melspectrogram(y=audio_padded, sr=16000, n_fft=1024, hop_length=1096, power=2)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)  # mels x time
        log_melspec = log_melspec.T
        # print('log_melspec',log_melspec.shape)#hop_length=512~(n_mels, 88)/hop_length=160~(34, 281)/256～(34, 176)/128～(34, 351)#(192,34,128)
        # print(type(log_melspec))#<class 'numpy.ndarray'>
        ##############

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        # to tensors
        word_seq_tensor = words_to_tensor(self.lang_model, word_seq, sample_end_time)
        extended_word_seq, text_token_padded, text = extend_word_seq(self.lang_model, word_seq, sample_end_time)
        vec_seq = torch.from_numpy(vec_seq).reshape((vec_seq.shape[0], -1)).float()
        pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
        audio_padded = torch.from_numpy(audio_padded).float()
        log_melspec = torch.from_numpy(log_melspec).float()
        spectrogram = torch.from_numpy(spectrogram)

        # #####
        # # 解码ID回到token
        # tokens = self.tokenizer.convert_ids_to_tokens(text_token_padded)
        # # 将token转换回文本
        # decoded_text = self.tokenizer.convert_tokens_to_string(tokens)
        # print('decoded text',decoded_text)
        # print('original text',text)
        # #####

        return word_seq_tensor, extended_word_seq, text_token_padded, text, pose_seq, vec_seq, audio_padded, log_melspec, spectrogram, aux_info

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model

    def _make_speaker_model(self, lmdb_dir, cache_path):
        logging.info('  building a speaker model...')
        speaker_model = Vocab('vid', insert_default_tokens=False)

        lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        txn = lmdb_env.begin(write=False)
        cursor = txn.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            vid = video['vid']
            speaker_model.index_word(vid)

        lmdb_env.close()
        logging.info('    indexed %d videos' % speaker_model.n_words)
        self.speaker_model = speaker_model

        # cache
        with open(cache_path, 'wb') as f:
            pickle.dump(self.speaker_model, f)

