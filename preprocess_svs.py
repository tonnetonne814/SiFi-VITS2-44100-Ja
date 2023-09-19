import os
import torchaudio
import torch

from tqdm import tqdm
import random
import argparse
from f0_extractor import F0_extractor
import subprocess
from singDB_loader import get_annotated_data, get_lab_info

target_sr = 44100
hop_size = 512
song_min_s = 5000 
split_ratio = 0.005

def preprocess(folder, f0_method, out_dir, audio_norm):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    song_dir = os.path.join(folder, "DATABASE")
    oto2lab_path = os.path.join(folder,"kana2phonemes_002_oto2lab.table" )
    pitch_extractor = F0_extractor(samplerate=target_sr,
                                   hop_size=hop_size,
                                   device=device)
    count = 0
    filelist = list()
    for song_name in tqdm(os.listdir(song_dir), desc="Preprocessing..."):
        lab_path        = os.path.join(song_dir, song_name, song_name + ".lab")
        ust_path        = os.path.join(song_dir, song_name, song_name + ".ust")

        if audio_norm is True:
            orig_wav_path = os.path.join(song_dir, song_name, song_name + ".wav")
            norm_wav_path = os.path.join(song_dir, song_name, song_name + "_norm.wav")
            audio_norm_process(orig_wav_path, norm_wav_path)
            wav_path = norm_wav_path
        else:
            wav_path = os.path.join(song_dir, song_name, song_name + ".wav")

        anno_datalist = get_annotated_data(ust_path = ust_path,
                                           lab_path=lab_path,
                                           ono2lab_table_path=oto2lab_path)
        
        f0, vuv = pitch_extractor.compute_f0(path=wav_path, f0_method=f0_method)
        wav, sr = torchaudio.load(wav_path)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        if wav.size(0) > 1:
            print("[ERROR] Audio Channel is not 1ch.")
            exit()
        wav = torch.squeeze(wav, dim=0)
        
        t_s = 0
        t_e = 0
        ph_list             = list()
        ph_s_ms_list        = list()
        ph_e_ms_list        = list()
        ph_dur_ms_list      = list()
        word_s_ms_list      = list()
        word_e_ms_list      = list()
        word_dur_ms_list    = list()
        word_list           = list()
        noteID_list         = list()
        notename_list       = list()
        notedur_list        = list()
        n_ph_in_a_word      = list()

        for idx , line in enumerate(anno_datalist):

            # 無音時　かつ　長さが足りている時（開始時終了時にpauは来ない方式
            if line["ph"][0][0] == "pau":
                t_e = line["start_ms"]

                if idx == 0:
                    t_s = line["end_ms"]
                    continue
                
                # 保存データが短すぎる、かつ今回のpau_durも短いなら、再ループへ
                if t_e - t_s < song_min_s and line["dur_ms"] < song_min_s:
                    for data in line["ph"]:
                        ph_list             .append(data[0])
                        ph_s_ms_list        .append(data[1] - t_s)
                        ph_e_ms_list        .append(data[2] - t_s)
                        ph_dur_ms_list      .append(data[3])
                    word_s_ms_list      .append(line["start_ms"] - t_s)
                    word_e_ms_list      .append(line["end_ms"]   - t_s)
                    word_dur_ms_list    .append(line["dur_ms"])
                    word_list           .append(line["word"])
                    noteID_list         .append(line["note_num"])
                    notename_list       .append(line["notename"])
                    notedur_list        .append(line["note_dur"])
                    n_ph_in_a_word      .append(len(line["ph"]))
                    continue
                
                # check length
                word_len = len(word_list)
                if  word_len != len(word_s_ms_list) or \
                    word_len != len(word_e_ms_list) or \
                    word_len != len(word_dur_ms_list) or \
                    word_len != len(noteID_list) or \
                    word_len != len(notename_list) or \
                    word_len != len(n_ph_in_a_word):
                    print("[ERROR] word_length error. check preprocess.py")
                    exit()
                ph_len = len(ph_list)
                if ph_len != len(ph_s_ms_list) or \
                    ph_len != len(ph_e_ms_list) or \
                    ph_len != len(ph_dur_ms_list) or \
                    ph_len != int(torch.sum(torch.tensor(n_ph_in_a_word, dtype=torch.int64))):
                    print("[ERROR] ph_length error. check preprocess.py")
                    exit()

                # save
                basename = os.path.join(out_dir, str(count).zfill(5))
                w_s_idx = int((t_s/1000) * target_sr)  
                w_e_idx = int((t_e/1000) * target_sr)  
                h_s_idx = int((t_s/1000) * target_sr / hop_size)
                h_e_idx = int((t_e/1000) * target_sr / hop_size)

                # last check (diff is below 3 frame)
                diff = abs(abs(t_e -t_s) - int(torch.sum(torch.tensor(word_dur_ms_list, dtype=torch.int64))))
                if diff > 3 *(hop_size/target_sr) * 1000:
                    print("[WARNING] Detects errors of 3 frames or more. {:.5f} [ms]".format(diff))

                torchaudio.save(filepath=basename + ".wav", 
                                src=torch.unsqueeze(wav[w_s_idx:w_e_idx], dim=0) ,
                                sample_rate=target_sr)
                torch.save(f0[h_s_idx:h_e_idx], basename + "_f0.pt")
                torch.save(vuv[h_s_idx:h_e_idx],basename + "_vuv.pt")
                torch.save(ph_list,             basename + "_ph.pt" )
                torch.save(ph_s_ms_list,        basename + "_ph_s_ms.pt" )
                torch.save(ph_e_ms_list,        basename + "_ph_e_ms.pt" )
                torch.save(ph_dur_ms_list,      basename + "_ph_dur_ms.pt" )
                torch.save(word_s_ms_list,      basename + "_word_s_ms.pt" )
                torch.save(word_e_ms_list,      basename + "_word_e_ms.pt" )
                torch.save(word_dur_ms_list,    basename + "_word_dur_ms.pt" )
                torch.save(word_list,           basename + "_word.pt" )
                torch.save(noteID_list,         basename + "_noteID.pt" )
                torch.save(notename_list,       basename + "_notename.pt" )
                torch.save(notedur_list,        basename + "_notedur.pt" )
                torch.save(n_ph_in_a_word,      basename + "_n_ph_in_a_word.pt" )
                filelist.append(basename + "\n")
                count+=1 

                # init
                ph_list             = list()
                word_s_ms_list      = list()
                word_e_ms_list      = list()
                word_dur_ms_list    = list()
                word_list           = list()
                noteID_list         = list()
                notename_list       = list()
                notedur_list        = list()
                ph_s_ms_list        = list()
                ph_e_ms_list        = list()
                ph_dur_ms_list      = list()
                n_ph_in_a_word      = list()
                
                # next start time
                t_s = line["end_ms"]

            else:
                for data in line["ph"]:
                    ph_list             .append(data[0])
                    ph_s_ms_list        .append(data[1] - t_s)
                    ph_e_ms_list        .append(data[2] - t_s)
                    ph_dur_ms_list      .append(data[3])
                word_s_ms_list      .append(line["start_ms"] - t_s)
                word_e_ms_list      .append(line["end_ms"]   - t_s)
                word_dur_ms_list    .append(line["dur_ms"])
                word_list           .append(line["word"])
                noteID_list         .append(line["note_num"])
                notename_list       .append(line["notename"])
                notedur_list        .append(line["note_dur"])
                n_ph_in_a_word      .append(len(line["ph"]))

        if audio_norm is True:
            os.remove(wav_path)

    filelist_split(filelist=filelist)

def audio_norm_process(in_path, out_path):
    if os.path.isfile(in_path) is True:
        pass
    else:
        print("[ERROR] File is not existed : ", in_path)
        exit()
    cmd = ["ffmpeg-normalize", in_path,   "-o",  out_path]
    subprocess.run(cmd, encoding='utf-8', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def filelist_split(filelist):
    os.makedirs("./filelists/",exist_ok=True)   
    max_n = len(filelist)
    test_list = list()
    for _ in range(int(max_n * split_ratio)):
        n = len(filelist)
        idx = random.randint(9, int(n-1))
        txt = filelist.pop(idx)
        test_list.append(txt)

    max_n = len(filelist)
    val_list = list()
    for _ in range(int(max_n * split_ratio)):
        n = len(filelist)
        idx = random.randint(9, int(n-1))
        txt = filelist.pop(idx)
        val_list.append(txt)

    write_txt(f"./filelists/SVS_train_{target_sr}.txt", sorted(filelist))
    write_txt(f"./filelists/SVS_val_{target_sr}.txt",   sorted(val_list))
    write_txt(f"./filelists/SVS_test_{target_sr}.txt",  sorted(test_list))

    return 0

def read_txt(path):
    with open(path, mode="r", encoding="utf-8")as f:
        lines = f.readlines()
    return lines

def write_txt(path, lines):
    with open(path, mode="w", encoding="utf-8")as f:
        f.writelines(lines)

import glob 
def generate_word_list(dataset_dir, out_path):
    pathlist = glob.glob(f"{dataset_dir}/*.wav")
    word_list = list()
    for path in pathlist:
        word_path = path.replace(".wav", "_word.pt")
        word = torch.load(word_path)
        word_list.extend(word)
    word_list = sorted(set(word_list))
    torch.save(word_list, out_path)
    print(f"[INFO] Word list is saved at {out_path}")
    print(f"[INFO] Word list : {word_list}")
    return 0

def take_ph_statistics(dataset_dir, out_path, threshold_ms=50):
    pathlist = glob.glob(f"{dataset_dir}/**/*.lab", recursive=True)

    ph_list = list()
    for path in pathlist:
        for data in get_lab_info(path):
            ph_list.append(data["ph"])
    ph_to_id = {s.replace("\n", ""): i for i, s in enumerate(sorted(set(ph_list)))}
    id_to_ph = {i: s.replace("\n", "") for i, s in enumerate(sorted(set(ph_list)))}
    data_lists = [[] for _ in range(len(ph_to_id))]
    for path in pathlist:
        for data in get_lab_info(path):
            ph = data["ph"]
            ms = data["dur_ms"]
            data_lists[ph_to_id[ph]].append(ms)

    import numpy as np
    statistic_dict = {}
    for idx in range(len(ph_to_id)):
        data = np.array(data_lists[idx])
        ph = id_to_ph[idx]
        mean = np.mean(data)
        std = np.std(data)
        # 大体同じような値の奴は、そのまま使う
        if std<threshold_ms:
            statistic_dict[ph] = [mean, std]

    torch.save(statistic_dict, out_path)
    print(f"[INFO] ph duration statstics is saved at {out_path}")
    print(f"[INFO] ph statstics : {statistic_dict}")
    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--song_db_path',
                        type=str,
                        #required=True, 
                        default="./「波音リツ」歌声データベースVer2/")
    
    parser.add_argument('--f0_method',
                        type=str,
                        #required=True,
                        default="crepe")
    
    parser.add_argument('--dataset_out_dir',
                        type=str,
                        #required=True,
                        default="./dataset_SVS/")
    
    parser.add_argument('--audio_normalize',
                        type=str,
                        #required=True,
                        default=True)

    args = parser.parse_args()

    preprocess(folder= args.song_db_path,
               f0_method=args.f0_method,
               out_dir=args.dataset_out_dir,
               audio_norm=args.audio_normalize)
    
    take_ph_statistics(dataset_dir=args.song_db_path,
                       out_path=os.path.join(args.dataset_out_dir, "ph_statistics.pt"))

    #import glob
    #filelist = list()
    #for path in glob.glob("./dataset_SVS/*.wav"):
    #    filelist.append(str(path).replace(".wav", "") + "\n")
    #filelist_split(filelist)
    