import os
import json
import argparse

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from lpmc.music_captioning.model.bart import BartCaptionModel
from lpmc.utils.eval_utils import load_pretrained
from lpmc.utils.audio_utils import load_audio, STR_CH_FIRST
from omegaconf import OmegaConf


def write_caption(caption_dict:dict[int, str], dest_path:str, cap_folder:str):
    if not os.path.isdir(cap_folder):
        os.mkdir(cap_folder)

    try:
        with open(dest_path, 'w') as f:
            json.dump(caption_dict, f, indent=4)
    except Exception as e:
        print(f"FAILED WRITING: {dest_path}")

def get_audio(audio_path, duration=10, target_sr=16000):
    n_samples = int(duration * target_sr)
    audio, sr = load_audio(
        path= audio_path,
        ch_format= STR_CH_FIRST,
        sample_rate= target_sr,
        downmix_to_mono= True,
    )
    if len(audio.shape) == 2:
        audio = audio.mean(0, False)  # to mono
    input_size = int(n_samples)
    if audio.shape[-1] < input_size:  # pad sequence
        pad = np.zeros(input_size)
        pad[: audio.shape[-1]] = audio
        audio = pad
    ceil = int(audio.shape[-1] // n_samples)
    audio = torch.from_numpy(np.stack(np.split(audio[:ceil * n_samples], ceil)).astype('float32'))
    return audio
 
def captioning(args, captions_paths:list[tuple[str, str, str]]):
    save_dir = f"exp/{args.framework}/{args.caption_type}/"
    config = OmegaConf.load(os.path.join(save_dir, "hparams.yaml"))
    model = BartCaptionModel(max_length = config.max_length)
    model, _ = load_pretrained(args, save_dir, model, mdp=config.multiprocessing_distributed)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    model.eval()

    for sdtk_file, cap_path, cap_folder in tqdm(captions_paths):
        try:
            audio_tensor = get_audio(audio_path=sdtk_file)
            audio_tensor = audio_tensor.cuda(args.gpu, non_blocking=True)
        except Exception as e:
            print(f"FAILED TO READ: {sdtk_file}")
            continue

        with torch.no_grad():
            output = model.generate(
                samples=audio_tensor,
                num_beams=args.num_beams,
            )

        inference = {}
        chunks_indexes = range(audio_tensor.shape[0])
        for chunk_idx, text in zip(chunks_indexes, output):
            time = f"{chunk_idx * 10}:00-{(chunk_idx + 1) * 10}:00"
            item = {"text":text,"time":time}
            inference[chunk_idx] = item

            write_caption(inference, cap_path, cap_folder)

def get_captions_paths(dataset_folder) -> list[tuple[str, str, str]]:
    files = []
    skiped = 0
    for game_folder in sorted(os.listdir(dataset_folder)):
        soundtracks_folder = os.path.join(dataset_folder, game_folder, 'soundtracks')
        music_caps_folder = os.path.join(dataset_folder, game_folder, 'music_caps')

        for soundtrack_file in sorted(os.listdir(soundtracks_folder)):
            soundtrack_path = os.path.join(soundtracks_folder, soundtrack_file)
            music_caps_file_path = os.path.join(music_caps_folder, soundtrack_file[:-3]+'json')

            if not os.path.exists(music_caps_file_path):
                files.append((soundtrack_path, music_caps_file_path, music_caps_folder))
            else:
                print(f"SKIPING {soundtrack_path}")
                skiped += 1

    return files

def main():
    parser = argparse.ArgumentParser(description='VMDB Captioning')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument("--framework", default="transfer", type=str)
    parser.add_argument("--caption_type", default="lp_music_caps", type=str)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--num_beams", default=5, type=int)
    parser.add_argument("--model_type", default="last", type=str)
    parser.add_argument('--dataset_root', type=str, default="/home/es119256/dados/datasets/vmdb_3/nintendo-snes-spc", help="path for the dataset games folder")

    args = parser.parse_args()

    captions_paths = get_captions_paths(args.dataset_root)
    captioning(args, captions_paths)

if __name__ == '__main__':
    main()

    