import os
from tqdm import tqdm
import random
import argparse
import torchaudio 
from f0_extractor import F0_extractor
import torch

target_sr = 44100
hop_size = 512
song_min_s = 5000 # ms
split_ratio = 0.005

device =torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
os.makedirs("./filelists/",exist_ok=True)

def jsut_preprocess(dataset_dir:str = "./jsut_ver1.1/basic5000/", 
                    f0_method:str = "crepe",
                    results_folder="./dataset_TTS/jsut/",
                     normalize = False ):
    
    os.makedirs(results_folder, exist_ok=True)
    
    pitch_extractor = F0_extractor(samplerate=target_sr,
                                   hop_size=hop_size,
                                   device=device)

    wav_dir = os.path.join(dataset_dir, "wav")
    #"""
    for filename in tqdm(os.listdir(wav_dir)):
        wav_path = os.path.join(wav_dir, filename)
        if normalize is True:
            out_path = wav_path.replace(".wav", "_norm.wav")
            audio_norm_process(wav_path, out_path)
            wav_path = out_path

        y, sr = torchaudio.load(wav_path)
        y_converted = torchaudio.functional.resample(waveform=y, orig_freq=sr, new_freq=target_sr)
        save_path = os.path.join(results_folder, filename)
        torchaudio.save(save_path, y_converted, target_sr) 
        
        f0, vuv = pitch_extractor.compute_f0(path=save_path, f0_method=f0_method)
        torch.save(f0, save_path.replace(".wav", "_f0.pt") )
        torch.save(vuv, save_path.replace(".wav", "_vuv.pt") )
        if normalize is True:
            os.remove(wav_path)
    #"""
    txt_path = os.path.join(dataset_dir, "transcript_utf8.txt")
    text_list = list()
    for txt in read_txt(txt_path):
        if txt == "\n":
            continue
        name, sentence = txt.split(":")
        sentence = sentence.replace("\n", "")
        wav_filepath = os.path.join(results_folder, name )
        out_txt = wav_filepath + "|" + sentence + "\n"
        text_list.append(out_txt)

    max_n = len(text_list)
    test_list = list()
    for _ in range(int(max_n * split_ratio)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        test_list.append(txt)

    max_n = len(text_list)
    val_list = list()
    for _ in range(int(max_n * split_ratio)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        val_list.append(txt)

    write_txt(f"./filelists/TTS_jsut_train_{target_sr}.txt", text_list)
    write_txt(f"./filelists/TTS_jsut_val_{target_sr}.txt", val_list)
    write_txt(f"./filelists/TTS_jsut_test_{target_sr}.txt", test_list)
    
    return 0

def ita_preprocess(dataset_dir:str = "./path/to/ita_corpus", 
                    f0_method:str = "crepe",results_folder="./dataset_TTS/ita/", normalize=False):
    
    os.makedirs(results_folder, exist_ok=True)
    pitch_extractor = F0_extractor(samplerate=target_sr,
                                   hop_size=hop_size,
                                   device=device)
    
    folder_list = ["recitation", "emotion"]
    #"""
    for folder in folder_list:
        wav_dir = os.path.join(dataset_dir, folder)
        filelist = os.listdir(wav_dir)
        results_folder_dir = os.path.join(results_folder,folder)
        os.makedirs(results_folder_dir, exist_ok=True)
        
        for filename in tqdm(filelist):
            wav_path = os.path.join(wav_dir, filename)
            if normalize is True:
                out_path = wav_path.replace(".wav", "_norm.wav")
                audio_norm_process(wav_path, out_path)
                wav_path = out_path
            y, sr = torchaudio.load(wav_path)
            y_converted = torchaudio.functional.resample(waveform=y, orig_freq=sr, new_freq=target_sr)
            for idx in range(999):
                if str(idx).zfill(3) in filename:
                    break
                
            if folder == "recitation":  
                filename_out = "RECITATION324_" + str(idx).zfill(3) + ".wav"
            elif folder == "emotion":
                filename_out = "EMOTION100_"+ str(idx).zfill(3) + ".wav"
            else:
                print("ERROR. Check ITA corpus.")
                continue
            
            save_path = os.path.join(results_folder_dir, filename_out)
            torchaudio.save(save_path, y_converted, target_sr) 

            f0, vuv = pitch_extractor.compute_f0(path=save_path, f0_method=f0_method)
            torch.save(f0, save_path.replace(".wav", "_f0.pt") )
            torch.save(vuv, save_path.replace(".wav", "_vuv.pt") )
            
            if normalize is True:
                os.remove(wav_path)

    #"""
    txt_path = os.path.join(results_folder, "transcript_utf8.txt")
    text_list = list()
    for txt in read_txt(txt_path):
        if txt == "\n":
            continue
        name, sentence = txt.split(":")
        sentence, kana = sentence.split(",")
        sentence = sentence.replace("\n", "")
        if "RECITATION" in name:
            wav_filepath = os.path.join(results_folder,"recitation", name )
        elif "EMOTION" in name:
            wav_filepath = os.path.join(results_folder,"emotion", name )
            
        out_txt = wav_filepath + "|" + sentence + "\n"
        text_list.append(out_txt)

    max_n = len(text_list)
    test_list = list()
    for _ in range(int(max_n * split_ratio)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        test_list.append(txt)

    max_n = len(text_list)
    val_list = list()
    for _ in range(int(max_n * split_ratio)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        val_list.append(txt)

    write_txt(f"./filelists/TTS_ita_train_{target_sr}.txt", text_list)
    write_txt(f"./filelists/TTS_ita_val_{target_sr}.txt", val_list)
    write_txt(f"./filelists/TTS_ita_test_{target_sr}.txt", test_list)
    
    return 0

def homebrew_preprocess(dataset_dir:str = "./homebrew/", 
                    f0_method:str = "crepe",results_folder="./dataset_TTS/homebrew/", normalize=False ):
 
    os.makedirs(results_folder, exist_ok=True)
    pitch_extractor = F0_extractor(samplerate=target_sr,
                                   hop_size=hop_size,
                                   device=device)

    wav_dir = dataset_dir
    for filename in tqdm(os.listdir(wav_dir)):
        wav_path = os.path.join(wav_dir, filename)
        
        if normalize is True:
            out_path = wav_path.replace(".wav", "_norm.wav")
            audio_norm_process(wav_path, out_path)
            wav_path = out_path
        y, sr = torchaudio.load(wav_path)
        y_converted = torchaudio.functional.resample(waveform=y, orig_freq=sr, new_freq=target_sr)
        save_path = os.path.join(results_folder, filename)
        torchaudio.save(save_path, y_converted, target_sr) 

        f0, vuv = pitch_extractor.compute_f0(path=save_path, f0_method=f0_method)
        torch.save(f0, save_path.replace(".wav", "_f0.pt") )
        torch.save(vuv, save_path.replace(".wav", "_vuv.pt") )
        if normalize is True:
            os.remove(wav_path)
        
    txt_path = os.path.join(results_folder, "transcript_utf8.txt")
    text_list = list()
    for txt in read_txt(txt_path):
        if txt == "\n":
            continue
        name, sentence = txt.split(":")
        sentence = sentence.replace("\n", "")
        wav_filepath = os.path.join(results_folder, name)
        out_txt = wav_filepath + "|" + sentence + "\n"
        text_list.append(out_txt)

    max_n = len(text_list)
    test_list = list()
    for _ in range(int(max_n * split_ratio)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        test_list.append(txt)

    max_n = len(text_list)
    val_list = list()
    for _ in range(int(max_n * split_ratio)):
        n = len(text_list)
        idx = random.randint(9, int(n-1))
        txt = text_list.pop(idx)
        val_list.append(txt)

    write_txt(f"./filelists/TTS_homebrew_train_{target_sr}.txt", text_list)
    write_txt(f"./filelists/TTS_homebrew_val_{target_sr}.txt", val_list)
    write_txt(f"./filelists/TTS_homebrew_test_{target_sr}.txt", test_list)

    return 0

def read_txt(path):
    with open(path, mode="r", encoding="utf-8")as f:
        lines = f.readlines()
    return lines

def write_txt(path, lines):
    with open(path, mode="w", encoding="utf-8")as f:
        f.writelines(lines)

import subprocess
def audio_norm_process(in_path, out_path):
    if os.path.isfile(in_path) is True:
        pass
    else:
        print("[ERROR] File is not existed : ", in_path)
        exit()
    cmd = ["ffmpeg-normalize", in_path,   "-o",  out_path]
    subprocess.run(cmd, encoding='utf-8', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name',
                        type=str,
                        #required=True, 
                        default="jsut",
                        help='jsut / ita / homebrew')
    parser.add_argument('--folder_path',
                        type=str,
                        #required=True, 
                        default="./jsut_ver1.1/basic5000/",
                        help='Path to corpus folder')
    parser.add_argument('--f0_method',
                        type=str,
                        #required=True,
                        default="crepe")
    parser.add_argument('--dataset_out_dir',
                        type=str,
                        #required=True,
                        default="./dataset_TTS/")
    parser.add_argument('--audio_normalize',
                        type=str,
                        #required=True,
                        default=True)

    
    args = parser.parse_args()
    dataset_out_dir = os.path.join(args.dataset_out_dir, args.dataset_name)

    if args.dataset_name == "jsut":
        jsut_preprocess(dataset_dir=args.folder_path, f0_method=args.f0_method, results_folder=dataset_out_dir, normalize=args.audio_normalize)
    elif args.dataset_name == "ita":
        ita_preprocess(dataset_dir=args.folder_path, f0_method=args.f0_method, results_folder=dataset_out_dir, normalize=args.audio_normalize)
    elif args.dataset_name == "homebrew":
        homebrew_preprocess(dataset_dir=args.folder_path, f0_method=args.f0_method, results_folder=dataset_out_dir, normalize=args.audio_normalize)
    else:
        print("ERROR. Check dataset_name.")
