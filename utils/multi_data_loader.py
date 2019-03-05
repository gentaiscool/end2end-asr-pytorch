import librosa
import json
import math
import numpy as np
import os
import scipy.signal
import torch

from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from utils import constant
from utils.audio import load_audio, get_audio_length, audio_with_sox, augment_audio_with_sox, load_randomly_augmented_audio


windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super(SpectrogramParser, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')

    def parse_audio(self, audio_path):
        if self.augment:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path)

        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)

        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)

        # Short-time Fourier transform (STFT)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)

        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)

        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError


class MultiSpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath_list, label2id, normalize=False, augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,/path/to/audio.txt
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        max_size = 0

        ids_list = []
        for manifest_filepath in manifest_filepath_list:
            with open(manifest_filepath) as f:
                ids = f.readlines()

            ids = [x.strip().split(',') for x in ids]
            ids_list.append(ids)

            max_size = max(max_size, len(ids))
        
        self.ids_list = ids_list
        self.size = max_size * len(manifest_filepath_list)
        self.label2id = label2id
        super(MultiSpectrogramDataset, self).__init__(
            audio_conf, normalize, augment)

    def __getitem__(self, index):
        # sample equally from datasets
        spect_list = []
        transcript_list = []
        for i in range(len(self.ids_list)):
            ids = self.ids_list[i]
            sample = ids[index % len(ids)]
            audio_path, transcript_path = sample[0], sample[1]
            spect = self.parse_audio(audio_path)[:,:constant.args.src_max_len]
            transcript = self.parse_transcript(transcript_path)
            spect_list.append(spect)
            transcript_list.append(transcript)

        return spect_list, transcript_list

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '').lower()

        transcript = list(
            filter(None, [self.label2id.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size


class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 sample_rate=16000,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = get_audio_length(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_end = noise_start + data_len
        noise_dst = audio_with_sox(
            noise_path, self.sample_rate, noise_start, noise_end)
        assert len(data) == len(noise_dst)
        noise_energy = np.sqrt(noise_dst.dot(noise_dst) / noise_dst.size)
        data_energy = np.sqrt(data.dot(data) / data.size)
        data += noise_level * noise_dst * data_energy / noise_energy
        return data


def _multi_collate_fn(batch):
    def func(p):
        input_data = p[0]
        max_seq_len = 0
        for i in range(len(input_data)):
            max_seq_len = max(max_seq_len, input_data[i].size(1))
        return max_seq_len

    def func_tgt(p):
        target_data = p[1]
        max_target_len = 0
        for i in range(len(target_data)):
            max_target_len = max(max_target_len, len(target_data[i]))
        return max_target_len

    # descendingly sorted
    num_dataset = len(batch[0])
    max_target = max(batch, key=func_tgt)[1]
    max_target_len = 0
    for i in range(len(max_target)):
        max_target_len = max(max_target_len, len(max_target[i]))

    max_seq_len = 0
    freq_len = 0
    max_seq = max(batch, key=func)[0]
    for i in range(len(max_seq)):
        max_seq_len = max(max_seq_len, max_seq[i].size(1))
        freq_len = max(freq_len, max_seq[i].size(0))

    inputs = torch.zeros(len(batch) * num_dataset, 1, freq_len, max_seq_len)
    input_sizes = torch.IntTensor(len(batch) * num_dataset)
    input_percentages = torch.FloatTensor(len(batch) * num_dataset)

    targets = torch.zeros(len(batch) * num_dataset, max_target_len).long()
    target_sizes = torch.IntTensor(len(batch) * num_dataset)
    
    sample_id = 0
    for x in range(len(batch)):
        sample = batch[x]
        input_data = sample[0]
        target = sample[1]
        for j in range(len(input_data)):
            seq_length = input_data[j].size(1)
            input_sizes[sample_id] = seq_length
            inputs[sample_id][0].narrow(1, 0, seq_length).copy_(input_data[j])
            input_percentages[sample_id] = seq_length / float(max_seq_len)
            target_sizes[sample_id] = len(target[j])
            targets[sample_id][:len(target[j])] = torch.IntTensor(target[j])
            sample_id += 1

    # sort
    sorted_input_sizes, indices = torch.sort(input_sizes, descending=True)
    sorted_inputs = inputs[indices]
    sorted_input_percentages = input_percentages[indices]
    sorted_target_sizes = target_sizes[indices]
    sorted_targets = targets[indices]
    
    return sorted_inputs, sorted_targets, sorted_input_percentages, sorted_input_sizes, sorted_target_sizes


class MultiAudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(MultiAudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _multi_collate_fn


class MultiBucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(MultiBucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        print("batch_size:", batch_size)
        self.bins = [ids[i:i + batch_size]
                     for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)
