import os
import argparse
import torch
import yaml
from models import VITS2_based_SiFiTTS
from singDB_loader import get_ust_info, get_g2p_dict_from_tabledata
import pyopenjtalk
import re
import jaconv
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch
from tqdm import tqdm

target_sr = 44100
hop_size = 512
song_min_s = 5000 # ms
split_ratio = 0.005

device =torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
os.makedirs("./filelists/",exist_ok=True)

def get_duration_from_MAS(hps):

    oto2lab, \
    ph_symbol_to_id, id_to_ph_symbol, \
    word_symbol_to_id,id_to_word_symbol = get_g2p_dict_from_tabledata(hps["oto2lab_path"], include_converter=True)

    vocab_size = len(ph_symbol_to_id) + 1

    net_g = VITS2_based_SiFiTTS(
                hps = hps,
                n_vocab=vocab_size,
                spec_channels=hps["spec_encoder"]["spec_channels"],
                segment_size=int(hps["segments_size"] // hps["hop_length"]),
                n_speakers=hps["common"]["n_speaker"],
                gin_channels=hps["common"]["gin_channels"],
                **hps["hifi_gan"]).cuda()    
    
    lines = []
    lines += read_txt(hps["train_data_path"])
    lines += read_txt(hps["eval_data_path"])
    lines += read_txt(hps["test_data_path"])
    count = 0
    for line in tqdm(lines, desc="get duration from MAS..."):
        count += 1
        if count < 4300:
            continue
        path, text = line.split("|")
        spec_path = path + "_spec.pt"
        if os.path.exists(spec_path):
            spec = torch.load(spec_path)
        else:
            wav_path = path + ".wav"
            audio_norm, sr = load_wav_to_torch(wav_path)
            assert sr == hps["sampling_rate"]

            spec = spectrogram_torch(audio_norm.unsqueeze(0), hps["filter_length"],
                hps["sampling_rate"], hps["hop_length"], hps["win_length"],
                center=False)
            spec = torch.squeeze(spec, 0)
            if spec.size(1) == -1:
                print("ERROR SPEC")
            torch.save(spec, spec_path)
        spec = spec.unsqueeze(0).cuda()

        spec_lengths = torch.tensor(spec.size(2), dtype=torch.int64).view(-1).cuda()
        ph_IDs = get_ph_ID(text, oto2lab, ph_symbol_to_id).view(1,-1).cuda()
        ph_IDs_lengths = torch.tensor(ph_IDs.size(1), dtype=torch.int64).view(-1).cuda()
        speakerID = torch.zeros(size=(1,), dtype=torch.int64).view(-1).cuda() # speaker id is fixed

        with torch.inference_mode():
            
            ph_dur, attn = net_g.get_mas_output(spec, 
                                                spec_lengths,
                                                ph_IDs, 
                                                ph_IDs_lengths,
                                                speakerID)
        
        torch.save(ph_dur[0][0].to('cpu').detach().numpy().copy(), path + "_ph_frame_dur.pt")
        torch.save(attn[0][0].to('cpu').detach().numpy().copy(),   path + "_ph_spec_attn.pt")

    return 0

def read_txt(path):
    with open(path, mode="r", encoding="utf-8")as f:
        lines = f.readlines()
    return lines

def write_txt(path, lines):
    with open(path, mode="w", encoding="utf-8")as f:
        f.writelines(lines)

def get_ph_ID(text, oto2lab, ph_symbol_to_id):
    small_letters = ["ぁ", "ぃ", "ぅ", "ぇ", "ぉ", 
                     "ァ", "ィ", "ゥ", "ェ", "ォ",
                     "ゃ", "ゅ", "ょ", "っ", "ゎ",
                     "ャ", "ュ", "ョ", "ッ", "ヮ"]
    find_desc = "|".join(small_letters)
   
    kana = pyopenjtalk.g2p(text, kana=True)
    hira = " ".join(jaconv.kata2hira(kana))
    hira_back= hira

    skip = 0
    for idx, m in enumerate(re.finditer(find_desc, hira)):
        pos = m.span()[0]-(idx-skip)-1 
        if pos < 0:
            skip += 1
            pass
        else:
            hira = hira[:pos] + hira[pos+1:]
    hira = hira.split(" ")
    ph_list = []
    for word in hira:
        try:
            # jsut base covert rule
            if word == "・" or word == "-" or word == "−" or word == "？" :
                continue
            if word == '．':
                word = "。"
            if word == '乃':
                word = "の"
            elif word == '珂':
                ph_list += [oto2lab["か"]]
                continue
            elif word == '是':
                ph_list += [oto2lab["ぜ"]]
                continue

            elif word == '哉':
                ph_list += [oto2lab["や"]]
                continue
            elif word == '孺':
                ph_list += [oto2lab["じゅ"]]
                continue

            if word == "吐":
                ph_list += [oto2lab["ば"]]
                ph_list += [oto2lab["き"]]
                continue
            elif word == '熏':
                ph_list += [oto2lab["く"]]
                ph_list += [oto2lab["ん"]]
                continue
            elif word == '撃':
                ph_list += [oto2lab["げ"]]
                ph_list += [oto2lab["き"]]
                continue
            elif word == '哨':
                ph_list += [oto2lab["しょ"]]
                ph_list += [oto2lab["う"]]
                continue
            elif word == '逓':
                ph_list += [oto2lab["て"]]
                ph_list += [oto2lab["い"]]
                continue
            elif word == '騎':
                ph_list += [oto2lab["き"]]
                continue
            elif word == '閃':
                ph_list += [oto2lab["せ"]]
                ph_list += [oto2lab["ん"]]
                continue
            elif word == '抽':
                ph_list += [oto2lab["ちゅ"]]
                ph_list += [oto2lab["う"]]
                continue
            elif word == '分':
                ph_list += [oto2lab["ぶ"]]
                ph_list += [oto2lab["ん"]]
                continue
            elif word == '斬':
                ph_list += [oto2lab["ざ"]]
                ph_list += [oto2lab["ん"]]
                continue
            elif word == '醸':
                ph_list += [oto2lab["じょ"]]
                ph_list += [oto2lab["う"]]
                continue
            elif word == '勃':
                ph_list += [oto2lab["ぼ"]]
                ph_list += [oto2lab["つ"]]
                continue
            elif word == '煖':
                ph_list += [oto2lab["え"]]
                ph_list += [oto2lab["ん"]]
                continue
            elif word == '四':
                ph_list += [oto2lab["し"]]
                continue
            elif word == '川':
                ph_list += [oto2lab["せ"]]
                ph_list += [oto2lab["ん"]]
                continue
            elif word == '省':
                ph_list += [oto2lab["しょ"]]
                ph_list += [oto2lab["う"]]
                continue
            elif word == '雅':
                ph_list += [oto2lab["が"]]
                continue
            elif word == '安':
                ph_list += [oto2lab["あ"]]
                ph_list += [oto2lab["ん"]]
                continue
            elif word == '県':
                ph_list += [oto2lab["け"]]
                ph_list += [oto2lab["ん"]]
                continue
            elif word == '高':
                ph_list += [oto2lab["こ"]]
                ph_list += [oto2lab["う"]]
                continue
            elif word == '頤':
                ph_list += [oto2lab["い"]]
                continue
            elif word == '墓':
                ph_list += [oto2lab["ぼ"]]
                continue
            elif word == '闕':
                ph_list += [oto2lab["け"]]
                ph_list += [oto2lab["つ"]]
                continue
            elif word == '窒':
                ph_list += [oto2lab["ち"]]
                ph_list += [oto2lab["つ"]]
                continue
            elif word == '菌':
                ph_list += [oto2lab["き"]]
                ph_list += [oto2lab["ん"]]
                continue
            elif word == '聚':
                ph_list += [oto2lab["じゅ"]]
                continue
            elif word == '慎':
                ph_list += [oto2lab["し"]]
                ph_list += [oto2lab["ん"]]
                continue
            elif word == '果':
                ph_list += [oto2lab["か"]]
                continue
            elif word == '胞':
                ph_list += [oto2lab["ほ"]]
                ph_list += [oto2lab["う"]]
                continue
            elif word == '溶':
                ph_list += [oto2lab["よ"]]
                ph_list += [oto2lab["う"]]
                continue
            elif word == '結':
                ph_list += [oto2lab["け"]]
                ph_list += [oto2lab["つ"]]
                continue
            elif word == '腔':
                ph_list += [oto2lab["こ"]]
                ph_list += [oto2lab["う"]]
                continue
            elif word == '療':
                ph_list += [oto2lab["りょ"]]
                ph_list += [oto2lab["う"]]
                continue
            elif word == '々': 
                ph_list += [oto2lab["し"]]
                continue
            elif word == '禕': 
                ph_list += [oto2lab["い"]]
                continue

            if word == 'ー':
                c_ph = ph_list[-1]
                if len(c_ph) != 1:
                    ph_list += c_ph[-1]
                else:
                    ph_list += c_ph
                continue

            elif word == "ーっ":
                c_ph = ph_list[-1]
                if len(c_ph) != 1:
                    ph_list += [c_ph[-1]+" "+ oto2lab["っ"]]
                else:
                    ph_list += [c_ph     + " " + oto2lab["っ"]]
                continue

            try:
                word_ph = oto2lab[word]
            except:
                txt = ""   
                word_ph = ""   
                # check cl
                for w in word:
                    if w == "っ":
                        word_ph += oto2lab[txt] + " " + oto2lab[w] # pull from dict
                        txt = "" #reset
                    else:
                        txt += w # stack
                # last pull
                if txt != "":
                    word_ph += oto2lab[txt]
            ph_list += [word_ph]
        except:
            tmp = []
            for w in word:
                ph_list += [oto2lab[w]]
                tmp += [oto2lab[w]]
            print(f"[WARNING] {word} is not existed in oto2lab.")
            print(f"[WARNING] Full text is {hira_back}.")
            print(f"[WARNING] {word} is converted to {tmp}.")

    sequence = []
    for symbols in ph_list:
        for symbol in symbols.split(" "):
            symbol_id = ph_symbol_to_id[symbol]
            sequence += [symbol_id]
    return torch.tensor(sequence, dtype=torch.int64)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config',
                        type=str,
                        default="./configs/config.yaml")
    
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, mode="r", encoding="utf-8") as f:
      hparams = yaml.safe_load(f)

    get_duration_from_MAS(hps=hparams)
