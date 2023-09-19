import time
import os
import random
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import torchaudio
from singDB_loader import get_g2p_dict_from_tabledata, get_g2p_dict_from_training_data
import logger, copy
import commons 
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from scipy.interpolate import interp1d
import commons 
from sifigan.utils import dilated_factor        
from sifigan.utils.features import SignalGenerator
import pyopenjtalk
import jaconv
import re

# データセット読み込み君本体
class TextAudioLoader_TTS_f0diff(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams):

        self.small_letters = ["ぁ", "ぃ", "ぅ", "ぇ", "ぉ", 
                              "ァ", "ィ", "ゥ", "ェ", "ォ",
                              "ゃ", "ゅ", "ょ", "っ", "ゎ",
                              "ャ", "ュ", "ョ", "ッ", "ヮ"]
        self.find_desc = "|".join(self.small_letters)

        self.basepath_list = load_filepaths_and_text(audiopaths_and_text, split="|")
        
        self.sampling_rate  = hparams["sampling_rate"]
        self.hop_length     = hparams["hop_length"] 
        self.filter_length  = hparams["filter_length"]
        self.win_length     = hparams["win_length"]
        self.wav_max_ms     = hparams["wav_max_ms"]
        self.wav_min_ms     = hparams["wav_min_ms"]
        self.f0_max         = hparams["f0_max"]

        self.oto2lab, self.ph_symbol_to_id,   self.id_to_ph_symbol, \
                      self.word_symbol_to_id,self.id_to_word_symbol = get_g2p_dict_from_tabledata(table_path=hparams["oto2lab_path"], include_converter=True)

        random.seed(hparams["seed"])
        random.shuffle(self.basepath_list) # ここでシャッフルしている
        self._filter()

    # 最小以下及び最大以上のデータを弾くフィルター関数
    def _filter(self):
        filtered_list = []
        lengths = []
        for basepath in tqdm(self.basepath_list, desc="Dataset Filtering..."):
            wav, sr = torchaudio.load(basepath[0]+".wav")
            ch, wav_len = wav.shape
            ms = wav_len / sr * 1000
            if self.wav_min_ms <= ms  and ms  <= self.wav_max_ms:
                filtered_list.append(basepath)
                lengths.append(wav_len // (2 * self.hop_length))
            else:
                print(f"EXCEEDED LENGTH : {basepath[0]}")
        self.basepath_list = filtered_list
        self.lengths = lengths
        
    def get_ph_vocab_size(self):
        return len(self.ph_symbol_to_id) + 1 # mask用

    def get_ph_ID(self, text, oto2lab, ph_symbol_to_id):
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

    def get_audio(self, filename):
        #print(filename)
        audio_norm, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", "_spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            if spec.size(1) == -1:
                print("ERROR SPEC")
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_item(self, basepath):

        filepath = basepath[0]
        text = basepath[1]

        # labのデータは推論時存在しない
        f0           = torch.load(filepath + "_f0.pt"          ) 
        ph_frame_dur = torch.load(filepath + "_ph_frame_dur.pt")
        speakerID    = 0 # 未実装 

        # tokenize and get duration
        ph_IDs              = self.get_ph_ID(text=text, oto2lab=self.oto2lab, ph_symbol_to_id=self.ph_symbol_to_id)
        f0_len              = len(f0)

        # padの影響で1ずれる。無問題と妄想
        if f0_len != int(torch.sum(torch.from_numpy(ph_frame_dur))):
            ph_frame_dur[-1] += 1 
            
        # 保障

        return (torch.tensor(f0          ,            dtype=torch.float32),          
                torch.tensor(ph_IDs,                    dtype=torch.int64)+1,        # maskを0とする。
                torch.tensor(ph_frame_dur,              dtype=torch.int64),   
                torch.tensor(speakerID,                 dtype=torch.int64)  )

    def __getitem__(self, index):
        return self.get_item(self.basepath_list[index])

    def __len__(self):
        return len(self.basepath_list)


class TextAudioCollate_TTS_f0diff():
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort( torch.LongTensor( [x[0].size(0) for x in batch] ),dim=0, descending=True )

        max_f0_len              = max([len(x[0]) for x in batch])
        #max_vuv_len            = max([len(x[1]) for x in batch])
        max_ph_IDs_len          = max([len(x[1]) for x in batch])
        max_ph_frame_dur_len    = max([len(x[2]) for x in batch])

        f0_lengths              = torch.LongTensor(len(batch))
        #vuv_lengths             = torch.LongTensor(len(batch))
        ph_IDs_lengths          = torch.LongTensor(len(batch))
        ph_frame_dur_lengths    = torch.LongTensor(len(batch))
        spkids                  = torch.LongTensor(len(batch))

        f0_padded               = torch.FloatTensor(len(batch), max_f0_len)
        #vuv_padded              = torch.LongTensor(len(batch),  max_vuv_len) 
        ph_IDs_padded           = torch.LongTensor(len(batch),   max_ph_IDs_len)
        ph_frame_dur_padded     = torch.LongTensor(len(batch),   max_ph_frame_dur_len)

        f0_padded.zero_()
        ph_IDs_padded.zero_()
        ph_frame_dur_padded.zero_()
        spkids.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            
            f0 = row[0]
            f0_padded[i, :f0.size(0)] = f0
            f0_lengths[i] = f0.size(0)

            #vuv = row[1]
            #vuv_padded[i, :vuv.size(0)] = vuv
            #vuv_lengths[i] = vuv.size(0)
            
            ph_IDs = row[1]
            ph_IDs_padded[i, :ph_IDs.size(0)] = ph_IDs
            ph_IDs_lengths[i] = ph_IDs.size(0) 
            
            ph_frame_dur = row[2]
            ph_frame_dur_padded[i,     :ph_frame_dur.size(0)] = ph_frame_dur
            ph_frame_dur_lengths[i] =   ph_frame_dur.size(0)
            
            spkids[i] = row[3]
            

        # 次元調整
        f0_padded = torch.unsqueeze(f0_padded, dim=1)  
           
        if self.return_ids:
            return  f0_padded,              f0_lengths,             \
                    ph_IDs_padded,          ph_IDs_lengths,         \
                    ph_frame_dur_padded,                            \
                    spkids,                                   \
                    ids_sorted_decreasing    
        
        return  f0_padded,              f0_lengths,             \
                ph_IDs_padded,          ph_IDs_lengths,         \
                ph_frame_dur_padded,                            \
                spkids


# データセット読み込み君本体
class TextAudioLoader_TTS_synth(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams):
        self.small_letters = ["ぁ", "ぃ", "ぅ", "ぇ", "ぉ", 
                              "ァ", "ィ", "ゥ", "ェ", "ォ",
                              "ゃ", "ゅ", "ょ", "っ", "ゎ",
                              "ャ", "ュ", "ョ", "ッ", "ヮ"]
        self.find_desc = "|".join(self.small_letters)

        self.basepath_list = load_filepaths_and_text(audiopaths_and_text, split="|")
        
        self.sampling_rate  = hparams["sampling_rate"]
        self.hop_length     = hparams["hop_length"] 
        self.filter_length  = hparams["filter_length"]
        self.win_length     = hparams["win_length"]
        self.wav_max_ms     = hparams["wav_max_ms"]
        self.wav_min_ms     = hparams["wav_min_ms"]
        self.f0_max         = hparams["f0_max"]

        self.df_f0_type = hparams["SiFiGAN_utils"]["df_f0_type"]
        self.dense_factors = hparams["SiFiGAN_utils"]["dense_factors"]
        self.upsample_scales= hparams["SiFiGANGenerator"]["upsample_scales"]

        # for SiFiGAN
        self.signal_generator = SignalGenerator(sample_rate=hparams["sampling_rate"],
                                                hop_size=hparams["hop_length"] ,
                                                sine_amp=hparams["SiFiGAN_utils"]["sine_amp"],
                                                noise_amp=hparams["SiFiGAN_utils"]["noise_amp"],
                                                signal_types=hparams["SiFiGAN_utils"]["signal_types"],)
        
        self.oto2lab, self.ph_symbol_to_id,   self.id_to_ph_symbol, \
                      self.word_symbol_to_id,self.id_to_word_symbol = get_g2p_dict_from_tabledata(table_path=hparams["oto2lab_path"], include_converter=True)
        random.seed(hparams["seed"])
        random.shuffle(self.basepath_list) # ここでシャッフルしている
        self._filter()

    # 最小以下及び最大以上のデータを弾くフィルター関数
    def _filter(self):
        filtered_list = []
        lengths = []
        for basepath in tqdm(self.basepath_list, desc="Dataset Filtering..."):
            wav, sr = torchaudio.load(basepath[0]+".wav")
            ch, wav_len = wav.shape
            ms = wav_len / sr * 1000
            if self.wav_min_ms <= ms  and ms  <= self.wav_max_ms:
                filtered_list.append(basepath)
                lengths.append(wav_len // (2 * self.hop_length))
            else:
                print(f"EXCEEDED LENGTH : {basepath}")
        self.basepath_list = filtered_list
        self.lengths = lengths
        
    def get_ph_vocab_size(self):
        return len(self.ph_symbol_to_id) + 1 # mask用

    def get_ph_ID(self, text):
        kana = pyopenjtalk.g2p(text, kana=True)
        hira = " ".join(jaconv.kata2hira(kana))
        for idx, m in enumerate(re.finditer(self.find_desc, hira)):
            pos = m.span()[0]-idx-1
            hira = hira[:pos] + hira[pos+1:]
        hira = hira.split(" ")

        ph_list = []
        for word in hira:
            if word == 'ー':
                c_ph = ph_list[-1]
                if len(c_ph) != 1:
                    ph_list += c_ph[-1]
                else:
                    ph_list += c_ph
                continue

            try:
                word_ph = self.oto2lab[word]
            except:
                txt = ""   
                word_ph = ""   
                # check cl
                for w in word:
                    if w == "っ":
                        word_ph += self.oto2lab[txt] + " " + self.oto2lab[w] # pull from dict
                        txt = "" #reset
                    else:
                        txt += w # stack
                # last pull
                if txt != "":
                    word_ph += self.oto2lab[txt]
            ph_list += [word_ph]
        sequence = []
        for symbols in ph_list:
            for symbol in symbols.split(" "):
                symbol_id = self.ph_symbol_to_id[symbol]
                sequence += [symbol_id]
        return torch.tensor(sequence, dtype=torch.int64)

    def get_audio(self, filename):
        #print(filename)
        audio_norm, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", "_spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            if spec.size(1) == -1:
                print("ERROR SPEC")
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_continuos_f0(self, f0):
        """Convert F0 to continuous F0
        Args:
            f0 (ndarray): original f0 sequence with the shape (T)
        Return:
            (ndarray): continuous f0 with the shape (T)
        """
        # get uv information as binary
        uv = np.float32(f0 != 0)
        # get start and end of f0
        if (f0 == 0).all():
            logger.warn("all of the f0 values are 0.")
            return uv, f0, False
        start_f0 = f0[f0 != 0][0]
        end_f0 = f0[f0 != 0][-1]
        # padding start and end of f0 sequence
        cf0 = copy.deepcopy(f0)
        start_idx = np.where(cf0 == start_f0)[0][0]
        end_idx = np.where(cf0 == end_f0)[0][-1]
        cf0[:start_idx] = start_f0
        cf0[end_idx:] = end_f0
        # get non-zero frame index
        nz_frames = np.where(cf0 != 0)[0]
        # perform linear interpolation
        f = interp1d(nz_frames, cf0[nz_frames])
        cf0 = f(np.arange(0, cf0.shape[0]))

        return uv, cf0, True

    def get_item(self, basepath):
        filepath = basepath[0]
        text = basepath[1]

        # labのデータは推論時存在しない
        spec, wav   = self.get_audio(filename= filepath+".wav") 
        f0          = torch.load(filepath + "_f0.pt"          ) 
        vuv         = torch.load(filepath + "_vuv.pt"         ) 
        speakerID   = 0 # 未実装 
        
        ph_IDs = self.get_ph_ID(text)
        _, spec_len         = spec.shape

        # paddingの影響で、1長くなることがあるので、その対策
        if  spec_len != len(vuv) or  spec_len != len(f0):
            f0 = f0[:spec_len]
            vuv = vuv[:spec_len]

        _, c_f0, _ = self.get_continuos_f0(f0) # for SiFiGAN
        Sinewave = self.signal_generator(torch.tensor(c_f0, dtype=torch.float32).view(1,1,-1))# for SiFiGAN

        prod_upsample_scales = np.cumprod(self.upsample_scales)
        df_sample_rates = [self.sampling_rate / self.hop_length * s for s in prod_upsample_scales]
        dfs = []
        for df, us in zip(self.dense_factors, prod_upsample_scales):
            dfs += [
                np.repeat(dilated_factor(c_f0, self.sampling_rate, df), us)
                if self.df_f0_type == "cf0"
                else np.repeat(dilated_factor(c_f0, self.sampling_rate, df), us)
            ]

        # 保障
        assert spec_len == len(f0)
        assert spec_len == len(vuv)
        for i in range(len(self.dense_factors)):
            assert Sinewave.size(2) * df_sample_rates[i] == len(dfs[i]) * self.sampling_rate

        return (torch.tensor(wav,                       dtype=torch.float32),    
                torch.tensor(spec,                      dtype=torch.float32),    
                torch.tensor(f0,                        dtype=torch.float32),  # diffusion用に正規化  
                torch.tensor(ph_IDs,                    dtype=torch.int64)+1,    # maskを0とする。
                torch.tensor(c_f0,                      dtype=torch.float32),
                dfs ,
                torch.tensor(speakerID,                 dtype=torch.int64))

    def __getitem__(self, index):
        return self.get_item(self.basepath_list[index])

    def __len__(self):
        return len(self.basepath_list)


class TextAudioCollate_TTS_synth():
    def __init__(self, hparams, return_ids=False):
        self.return_ids = return_ids
        self.dense_factors = hparams["SiFiGAN_utils"]["dense_factors"]
        self.upsample_scales = hparams["SiFiGANGenerator"]["upsample_scales"]
        
        # for SiFiGAN
        self.signal_generator = SignalGenerator(sample_rate=hparams["sampling_rate"],
                                                hop_size=hparams["hop_length"] ,
                                                sine_amp=hparams["SiFiGAN_utils"]["sine_amp"],
                                                noise_amp=hparams["SiFiGAN_utils"]["noise_amp"],
                                                signal_types=hparams["SiFiGAN_utils"]["signal_types"],)

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort( torch.LongTensor( [x[1].size(1) for x in batch] ),dim=0, descending=True )

        max_wav_len                 = max([    x[0].size(1) for x in batch])
        max_spec_len                = max([    x[1].size(1) for x in batch])
        max_f0_len                  = max([len(x[2]) for x in batch])
        max_ph_IDs_len              = max([len(x[3]) for x in batch])
        max_c_f0_len                = max([len(x[4]) for x in batch])

        spec_lengths    = torch.LongTensor(len(batch))
        ph_IDs_lengths          = torch.LongTensor(len(batch))
        spkids                  = torch.LongTensor(len(batch))

        wav_padded  = torch.FloatTensor(len(batch), 1, max_wav_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        f0_padded   = torch.FloatTensor(len(batch), max_f0_len)
        ph_IDs_padded       = torch.LongTensor(len(batch),   max_ph_IDs_len)
        c_f0_padded   = torch.FloatTensor(len(batch), max_c_f0_len)

        wav_padded.zero_()
        spec_padded.zero_()
        f0_padded.zero_()
        ph_IDs_padded.zero_()
        c_f0_padded   .zero_()

        dfs_padded = []
        dfs_lengths = torch.LongTensor(len(batch), len(self.upsample_scales))
        lengths=max_c_f0_len
        for idx, scales in enumerate(self.upsample_scales):
            lengths *= scales
            dfs_padded.append(torch.FloatTensor(len(batch), 1, int(lengths)).zero_())

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            wav = row[0]
            wav_padded[i, :, :wav.size(1)] = wav

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            f0 = row[2]
            f0_padded[i, :f0.size(0)] = f0

            ph_IDs = row[3]
            ph_IDs_padded[i, :ph_IDs.size(0)] = ph_IDs
            ph_IDs_lengths[i] = ph_IDs.size(0) 
            
            c_f0 = row[4]
            c_f0_padded[i, :c_f0.size(0)] = c_f0
            
            dfs = row[5]
            for idx in range(len(self.upsample_scales)):
                df = dfs[idx]
                dfs_padded[idx][i, 0, :len(df)] = torch.tensor(df, dtype=torch.float32)
                dfs_lengths[i, idx] = len(df)
            
            spkids[i] = row[6]

        # 次元調整
        f0_padded = torch.unsqueeze(f0_padded, dim=1)  
            
        # バッチ単位で作成
        Sinewaves = self.signal_generator(c_f0_padded.unsqueeze(1)).float()

        if self.return_ids:
            return  wav_padded,\
                    spec_padded,            spec_lengths,           \
                    f0_padded,\
                    ph_IDs_padded,          ph_IDs_lengths,         \
                    dfs_padded, \
                    Sinewaves, \
                    spkids, \
                    ids_sorted_decreasing    
        
        return  wav_padded,\
                spec_padded,            spec_lengths,           \
                f0_padded,\
                ph_IDs_padded,          ph_IDs_lengths,         \
                dfs_padded, \
                Sinewaves, \
                spkids


# データセット読み込み君本体
class TextAudioLoader_SVS_synth(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams):

        self.basepath_list = load_filepaths_and_text(audiopaths_and_text, split="|")
        
        self.sampling_rate  = hparams["sampling_rate"]
        self.hop_length     = hparams["hop_length"] 
        self.filter_length  = hparams["filter_length"]
        self.win_length     = hparams["win_length"]
        self.wav_max_ms     = hparams["wav_max_ms"]
        self.wav_min_ms     = hparams["wav_min_ms"]
        self.f0_max         = hparams["f0_max"]

        self.df_f0_type = hparams["SiFiGAN_utils"]["df_f0_type"]
        self.dense_factors = hparams["SiFiGAN_utils"]["dense_factors"]
        self.upsample_scales= hparams["SiFiGANGenerator"]["upsample_scales"]

        # for SiFiGAN
        self.signal_generator = SignalGenerator(sample_rate=hparams["sampling_rate"],
                                                hop_size=hparams["hop_length"] ,
                                                sine_amp=hparams["SiFiGAN_utils"]["sine_amp"],
                                                noise_amp=hparams["SiFiGAN_utils"]["noise_amp"],
                                                signal_types=hparams["SiFiGAN_utils"]["signal_types"],)
        
        self.oto2lab, self.ph_symbol_to_id,   self.id_to_ph_symbol, \
                      self.word_symbol_to_id,self.id_to_word_symbol = get_g2p_dict_from_tabledata(table_path=hparams["oto2lab_path"], include_converter=True)
        random.seed(hparams["seed"])
        random.shuffle(self.basepath_list) # ここでシャッフルしている
        self._filter()

    # 最小以下及び最大以上のデータを弾くフィルター関数
    def _filter(self):
        filtered_list = []
        lengths = []
        for basepath in tqdm(self.basepath_list, desc="Dataset Filtering..."):
            wav, sr = torchaudio.load(basepath[0]+".wav")
            ch, wav_len = wav.shape
            ms = wav_len / sr * 1000
            if self.wav_min_ms <= ms  and ms  <= self.wav_max_ms:
                filtered_list.append(basepath[0])
                lengths.append(wav_len // (2 * self.hop_length))
            else:
                print(f"EXCEEDED LENGTH : {basepath[0]}")
        self.basepath_list = filtered_list
        self.lengths = lengths
        
    def get_ph_vocab_size(self):
        return len(self.ph_symbol_to_id) + 1 # mask用

    def get_ph_ID(self, ph_list):
        sequence = []
        for symbol in ph_list:
            symbol_id = self.ph_symbol_to_id[symbol]
            sequence += [symbol_id]
        return torch.tensor(sequence, dtype=torch.int64)

    def get_word_ID(self, word_list):
        sequence = []
        for symbol in word_list:
            symbol_id = self.word_symbol_to_id[symbol]
            sequence += [symbol_id]
        return torch.tensor(sequence, dtype=torch.int64)
    
    def expand_note_info(self, ph_IDs, noteID, note_dur, n_ph_pooling):
        ph_IDs_lengths = torch.tensor(ph_IDs.size(1), dtype=torch.int64)
        ph_IDs_mask = torch.unsqueeze(commons.sequence_mask(ph_IDs_lengths.view(1), ph_IDs.size(1)), 1).to(ph_IDs.dtype) # [B, 1, ph_len]
        noteID_lengths = torch.tensor(noteID.size(1), dtype=torch.int64)
        noteID_mask = torch.unsqueeze(commons.sequence_mask(noteID_lengths.view(1), noteID.size(1)), 1).to(noteID.dtype) # [B, 1, ph_len]

        attn_mask     = torch.unsqueeze(noteID_mask, 2) * torch.unsqueeze(ph_IDs_mask, -1)    # attn_mask = [B, 1, ph_len, note(word)_len]
        attn          = commons.generate_path(duration=torch.unsqueeze(n_ph_pooling,dim=1), mask=attn_mask)
        attn          = torch.squeeze(attn, dim=1).permute(0,2,1).float()                             # attn=[Batch, note_len,] 
        # expand
        noteID        = torch.matmul(noteID.float().unsqueeze(1), attn)                                            # to [Batch, inner_channel, ph_len] 
        note_dur      = torch.matmul(note_dur.float().unsqueeze(1), attn)                     # to [Batch, inner_channel, ph_len] 

        return noteID.view(-1), note_dur.view(-1)

    def get_audio(self, filename):
        #print(filename)
        audio_norm, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", "_spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            if spec.size(1) == -1:
                print("ERROR SPEC")
            torch.save(spec, spec_filename)
        return spec, audio_norm
    
    def get_dur_frame_from_e_ms(self, e_ms):
        e_ms = torch.tensor(e_ms,dtype=torch.float32)
        #frames = torch.ceil(  (e_ms/1000)*self.sampling_rate / self.hop_length )   # 切り上げ
        frames = torch.floor(  (e_ms / 1000)*self.sampling_rate / self.hop_length )   # 切り捨て
        frames = torch.diff(frames, dim=0, prepend=frames.new_zeros(1))
        #for idx in reversed(range(len(frames))):
        #    if idx == 0:
        #        continue
        #    frames[idx] = frames[idx] - frames[idx-1]
        return torch.tensor(frames, dtype=torch.int64)
    
    def get_linspace_ph_id(self, ph_idx_in_a_word):
        for idx, n_ph in enumerate(ph_idx_in_a_word):
            if idx == 0:
                data = torch.linspace(start=int(n_ph-1), end=0, steps=int(n_ph), dtype=torch.int64)
            else:
                data = torch.concat([data, torch.linspace(start=int(n_ph-1), end=0, steps=int(n_ph), dtype=torch.int64)], dim=0)
        return data
    
    def get_frame_level_word_id(self, wordID, word_dur):
        #for idx, (id, dur) in enumerate(zip(wordID, word_dur)):
        #    if idx == 0:
        #        data = torch.ones(size=(dur,),dtype=torch.int64) * id
        #    else:
        #        data = torch.concat([data, torch.ones(size=(dur,),dtype=torch.int64) * id], dim=0)
        for idx, (id, dur) in enumerate(zip(wordID, word_dur)):
            if idx == 0:
                data = torch.ones(size=(dur,),dtype=torch.int64) * idx
            else:
                data = torch.concat([data, torch.ones(size=(dur,),dtype=torch.int64) * idx], dim=0)
        return data

    def get_continuos_f0(self, f0):
        """Convert F0 to continuous F0
        Args:
            f0 (ndarray): original f0 sequence with the shape (T)
        Return:
            (ndarray): continuous f0 with the shape (T)
        """
        # get uv information as binary
        uv = np.float32(f0 != 0)
        # get start and end of f0
        if (f0 == 0).all():
            logger.warn("all of the f0 values are 0.")
            return uv, f0, False
        start_f0 = f0[f0 != 0][0]
        end_f0 = f0[f0 != 0][-1]
        # padding start and end of f0 sequence
        cf0 = copy.deepcopy(f0)
        start_idx = np.where(cf0 == start_f0)[0][0]
        end_idx = np.where(cf0 == end_f0)[0][-1]
        cf0[:start_idx] = start_f0
        cf0[end_idx:] = end_f0
        # get non-zero frame index
        nz_frames = np.where(cf0 != 0)[0]
        # perform linear interpolation
        f = interp1d(nz_frames, cf0[nz_frames])
        cf0 = f(np.arange(0, cf0.shape[0]))

        return uv, cf0, True

    def get_ph_pooling_dur(self, ph_e, word_e):
        out = list()
        z_t = 0
        word_idx = 0
        for idx, e in enumerate(ph_e):
            idx += 1
            if word_e[word_idx] == e:
                out.append(int(idx - z_t))
                z_t = idx
                word_idx += 1
        return out

    def get_item(self, basepath):

        # labのデータは推論時存在しない
        spec, wav   = self.get_audio(filename= basepath+".wav") 
        f0          = torch.load(basepath + "_f0.pt"          )   
        word_dur_ms = torch.load(basepath + "_word_dur_ms.pt")
        speakerID   = 0 # 未実装 

        # tokenize and get duration
        ph_ids_path             = basepath + "_ph_ids.pt"
        ph_frame_dur_path       = basepath + "_ph_frame_dur.pt"
        word_frame_dur_path     = basepath + "_word_frame_dur.pt"
        ph_idx_in_a_word_path   = basepath + "_ph_idx_in_a_word.pt"
        n_ph_pooling_path       = basepath + "_n_ph_pooling.pt"
        note_ID_dur_path        = basepath + "_noteID_dur.pt"

        if os.path.exists(ph_ids_path) is True:
            ph_IDs = torch.load(ph_ids_path)
        else:
            ph = torch.load(basepath + "_ph.pt"          ) # ust or lab
            ph_IDs              = self.get_ph_ID(ph_list=ph)
            torch.save(ph_IDs, ph_ids_path)
        
        if os.path.exists(ph_frame_dur_path) is True:
            ph_frame_dur = torch.load(ph_frame_dur_path)
        else:
            ph_e_ms     = torch.load(basepath + "_ph_e_ms.pt"     )     # lab
            ph_frame_dur = self.get_dur_frame_from_e_ms(e_ms=ph_e_ms)
            torch.save(ph_frame_dur, ph_frame_dur_path)

        if os.path.exists(word_frame_dur_path) is True:
            word_frame_dur = torch.load(word_frame_dur_path)
        else:
            word_e_ms   = torch.load(basepath + "_word_e_ms.pt"   )     # lab
            word_frame_dur = self.get_dur_frame_from_e_ms(e_ms=word_e_ms)
            torch.save(word_frame_dur, word_frame_dur_path)
            
        if os.path.exists(ph_idx_in_a_word_path) is True:
            ph_idx_in_a_word = torch.load(ph_idx_in_a_word_path)
        else:
            ph_idx_in_a_word = torch.load(basepath + "_n_ph_in_a_word.pt" )   # lab
            ph_idx_in_a_word = self.get_linspace_ph_id(ph_idx_in_a_word=ph_idx_in_a_word)
            torch.save(ph_idx_in_a_word, ph_idx_in_a_word_path)

        if os.path.exists(n_ph_pooling_path) is True:
            n_ph_pooling = torch.load(n_ph_pooling_path)
        else:
            word_e_ms   = torch.load(basepath + "_word_e_ms.pt"   )     # lab
            ph_e_ms     = torch.load(basepath + "_ph_e_ms.pt"     )     # lab
            n_ph_pooling = self.get_ph_pooling_dur(ph_e=ph_e_ms, word_e=word_e_ms)
            torch.save(n_ph_pooling, n_ph_pooling_path)

        if os.path.exists(note_ID_dur_path) is True:
            note_ID_dur = torch.load(note_ID_dur_path)
            noteID, notedur = note_ID_dur
        else:
            noteID      = torch.load(basepath + "_noteID.pt"      ) # ust
            notedur     = torch.load(basepath + "_notedur.pt"     ) # ust    
            noteID, notedur     = self.expand_note_info(ph_IDs=ph_IDs.view(1,-1), 
                                                    noteID      =torch.tensor(noteID,dtype=torch.int64).view(1,-1),
                                                    note_dur    =torch.tensor(notedur,dtype=torch.int64).view(1,-1),
                                                    n_ph_pooling=torch.tensor(n_ph_pooling, dtype=torch.int64).view(1,-1))
            note_ID_dur = [noteID, notedur]
            torch.save(note_ID_dur, note_ID_dur_path)

        _, spec_len         = spec.shape
        
        # paddingの影響で、1長くなることがあるので、その対策
        if  spec_len != len(f0):
            #back = f0 # for debug
            f0 = f0[:spec_len]
        if spec_len != int(torch.sum(word_frame_dur)):
            word_frame_dur[-1] += 1 
        if spec_len != int(torch.sum(ph_frame_dur)):
            ph_frame_dur[-1] += 1 
        
        c_f0_path             = basepath+"_c_f0.pt"
        Sinewave_path             = basepath+"_Sinewave.pt"
        dfs_path             = basepath+"_dfs.pt"
        
        if os.path.exists(c_f0_path) is True:
            c_f0 = torch.load(c_f0_path)
        else:
            _, c_f0, _ = self.get_continuos_f0(f0) # for SiFiGAN
            torch.save(c_f0, c_f0_path)

        if os.path.exists(Sinewave_path) is True:
            Sinewave = torch.load(Sinewave_path)
        else:
            Sinewave = self.signal_generator(torch.tensor(c_f0, dtype=torch.float32).view(1,1,-1))# for SiFiGAN
            torch.save(Sinewave, Sinewave_path)

        if os.path.exists(Sinewave_path) is True:
            Sinewave = torch.load(Sinewave_path)
        else:
            Sinewave = self.signal_generator(torch.tensor(c_f0, dtype=torch.float32).view(1,1,-1))# for SiFiGAN
            torch.save(Sinewave, Sinewave_path)

        if os.path.exists(dfs_path) is True:
            dfs = torch.load(dfs_path)
        else:
            prod_upsample_scales = np.cumprod(self.upsample_scales)
            df_sample_rates = [self.sampling_rate / self.hop_length * s for s in prod_upsample_scales]
            dfs = []
            for df, us in zip(self.dense_factors, prod_upsample_scales):
                dfs += [
                    np.repeat(dilated_factor(c_f0, self.sampling_rate, df), us)
                    if self.df_f0_type == "cf0"
                    else np.repeat(dilated_factor(c_f0, self.sampling_rate, df), us)
                ]
            torch.save(dfs, dfs_path)
        
        # 保障 (Debug時のみ)
        # assert sum(n_ph_pooling) == len(ph_IDs)
        # assert int(torch.sum(word_frame_dur)) == int(torch.sum(ph_frame_dur))
        # assert len(ph_idx_in_a_word) == len(ph_IDs)
        # assert spec_len == int(torch.sum(word_frame_dur))
        # assert spec_len == len(f0)
        # assert len(ph_IDs) == len(noteID) # WORD ID が多い。おそらく"っ"の影響
        # for i in range(len(self.dense_factors)):
        #     assert Sinewave.size(2) * df_sample_rates[i] == len(dfs[i]) * self.sampling_rate

        return (torch.tensor(wav,                       dtype=torch.float32),    
                torch.tensor(spec,                      dtype=torch.float32),    
                torch.tensor(f0,                        dtype=torch.float32),  # diffusion用に正規化  
                torch.tensor(ph_IDs,                    dtype=torch.int64)+1,    # maskを0とする。
                torch.tensor(ph_frame_dur,              dtype=torch.int64),  
                torch.tensor(word_dur_ms,               dtype=torch.float32)/1000,  
                torch.tensor(word_frame_dur,            dtype=torch.int64),    
                torch.tensor(ph_idx_in_a_word,            dtype=torch.int64)+1,     # maskを0とする。
                torch.tensor(n_ph_pooling,              dtype=torch.int64),    # maskを0とする。
                torch.tensor(c_f0,                      dtype=torch.float32),
                dfs ,
                torch.tensor(speakerID,                 dtype=torch.int64))

    def __getitem__(self, index):
        return self.get_item(self.basepath_list[index])

    def __len__(self):
        return len(self.basepath_list)


class TextAudioCollate_SVS_synth():
    def __init__(self, hparams, return_ids=False):
        self.return_ids = return_ids
        self.dense_factors = hparams["SiFiGAN_utils"]["dense_factors"]
        self.upsample_scales = hparams["SiFiGANGenerator"]["upsample_scales"]
        
        # for SiFiGAN
        self.signal_generator = SignalGenerator(sample_rate=hparams["sampling_rate"],
                                                hop_size=hparams["hop_length"] ,
                                                sine_amp=hparams["SiFiGAN_utils"]["sine_amp"],
                                                noise_amp=hparams["SiFiGAN_utils"]["noise_amp"],
                                                signal_types=hparams["SiFiGAN_utils"]["signal_types"],)

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort( torch.LongTensor( [x[1].size(1) for x in batch] ),dim=0, descending=True )

        max_wav_len                 = max([    x[0].size(1) for x in batch])
        max_spec_len                = max([    x[1].size(1) for x in batch])
        max_f0_len                  = max([len(x[2]) for x in batch])
        max_ph_IDs_len              = max([len(x[3]) for x in batch])
        max_ph_frame_dur_len        = max([len(x[4]) for x in batch])
        max_word_dur_ms_len         = max([len(x[5]) for x in batch])
        max_word_frame_dur_len      = max([len(x[6]) for x in batch])
        max_ph_idx_in_a_word_len    = max([len(x[7]) for x in batch])
        max_n_ph_pooling_len        = max([len(x[8]) for x in batch])
        max_c_f0_len                = max([len(x[9]) for x in batch])

        spec_lengths    = torch.LongTensor(len(batch))
        ph_IDs_lengths          = torch.LongTensor(len(batch))
        word_frame_dur_lengths      = torch.LongTensor(len(batch))
        spkids                  = torch.LongTensor(len(batch))

        wav_padded  = torch.FloatTensor(len(batch), 1, max_wav_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        f0_padded   = torch.FloatTensor(len(batch), max_f0_len)
        ph_IDs_padded       = torch.LongTensor(len(batch),   max_ph_IDs_len)
        ph_frame_dur_padded = torch.LongTensor(len(batch),   max_ph_frame_dur_len)
        word_dur_ms_padded      = torch.FloatTensor(len(batch), max_word_dur_ms_len)
        word_frame_dur_padded   = torch.LongTensor(len(batch),  max_word_frame_dur_len)
        ph_idx_in_a_word_padded        = torch.LongTensor(len(batch), max_ph_idx_in_a_word_len)
        n_ph_pooling_padded          = torch.LongTensor(len(batch), max_n_ph_pooling_len)
        c_f0_padded   = torch.FloatTensor(len(batch), max_c_f0_len)

        wav_padded.zero_()
        spec_padded.zero_()
        f0_padded.zero_()
        ph_IDs_padded.zero_()
        ph_frame_dur_padded.zero_()
        word_dur_ms_padded.zero_()
        word_frame_dur_padded.zero_()
        ph_idx_in_a_word_padded.zero_()
        n_ph_pooling_padded.zero_()
        c_f0_padded   .zero_()

        dfs_padded = []
        dfs_lengths = torch.LongTensor(len(batch), len(self.upsample_scales))
        lengths=max_c_f0_len
        for idx, scales in enumerate(self.upsample_scales):
            lengths *= scales
            dfs_padded.append(torch.FloatTensor(len(batch), 1, int(lengths)).zero_())

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            wav = row[0]
            wav_padded[i, :, :wav.size(1)] = wav

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            f0 = row[2]
            f0_padded[i, :f0.size(0)] = f0

            ph_IDs = row[3]
            ph_IDs_padded[i, :ph_IDs.size(0)] = ph_IDs
            ph_IDs_lengths[i] = ph_IDs.size(0) 

            ph_frame_dur = row[4]
            ph_frame_dur_padded[i,     :ph_frame_dur.size(0)] = ph_frame_dur
            
            word_dur_ms = row[5]
            word_dur_ms_padded[i,     :word_dur_ms.size(0)] = word_dur_ms
            
            word_frame_dur = row[6]
            word_frame_dur_padded[i,     :word_frame_dur.size(0)] = word_frame_dur
            word_frame_dur_lengths[i] =   word_frame_dur.size(0)

            ph_idx_in_a_word = row[7]
            ph_idx_in_a_word_padded[i, :ph_idx_in_a_word.size(0)] = ph_idx_in_a_word
            
            n_ph_pooling = row[8]
            n_ph_pooling_padded[i, :n_ph_pooling.size(0)] = n_ph_pooling
            
            c_f0 = row[9]
            c_f0_padded[i, :c_f0.size(0)] = c_f0
            
            dfs = row[10]
            for idx in range(len(self.upsample_scales)):
                df = dfs[idx]
                dfs_padded[idx][i, 0, :len(df)] = torch.tensor(df, dtype=torch.float32)
                dfs_lengths[i, idx] = len(df)
            
            spkids[i] = row[11]

        # 次元調整
        f0_padded = torch.unsqueeze(f0_padded, dim=1)  
            
        # バッチ単位で作成
        Sinewaves = self.signal_generator(c_f0_padded.unsqueeze(1)).float()

        if self.return_ids:
            return  wav_padded,\
                    spec_padded,            spec_lengths,           \
                    f0_padded,\
                    ph_IDs_padded,          ph_IDs_lengths,         \
                    ph_frame_dur_padded,\
                    word_dur_ms_padded,     \
                    word_frame_dur_padded,  word_frame_dur_lengths, \
                    ph_idx_in_a_word_padded,  \
                    n_ph_pooling_padded, \
                    dfs_padded, \
                    Sinewaves, \
                    spkids, \
                    ids_sorted_decreasing    
        
        return  wav_padded,\
                spec_padded,            spec_lengths,           \
                f0_padded,\
                ph_IDs_padded,          ph_IDs_lengths,         \
                ph_frame_dur_padded,\
                word_dur_ms_padded,     \
                word_frame_dur_padded,  word_frame_dur_lengths, \
                ph_idx_in_a_word_padded,  \
                n_ph_pooling_padded, \
                dfs_padded, \
                Sinewaves, \
                spkids

# データセット読み込み君本体
class TextAudioLoader_SVS_f0diff(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams):

        self.basepath_list = load_filepaths_and_text(audiopaths_and_text, split="|")
        
        self.sampling_rate  = hparams["sampling_rate"]
        self.hop_length     = hparams["hop_length"] 
        self.filter_length  = hparams["filter_length"]
        self.win_length     = hparams["win_length"]
        self.wav_max_ms     = hparams["wav_max_ms"]
        self.wav_min_ms     = hparams["wav_min_ms"]
        self.f0_max         = hparams["f0_max"]

        self.oto2lab, self.ph_symbol_to_id,   self.id_to_ph_symbol, \
                      self.word_symbol_to_id,self.id_to_word_symbol = get_g2p_dict_from_tabledata(table_path=hparams["oto2lab_path"], include_converter=True)
        with open(hparams["noteid2hz_txt_path"], mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        self.id_to_hz = {}
        for idx, line in enumerate(lines):
            id, hz = line.split(",")

            self.id_to_hz[idx-1] = float(hz)

        random.seed(hparams["seed"])
        random.shuffle(self.basepath_list) # ここでシャッフルしている
        self._filter()

    # 最小以下及び最大以上のデータを弾くフィルター関数
    def _filter(self):
        filtered_list = []
        lengths = []
        for basepath in tqdm(self.basepath_list, desc="Dataset Filtering..."):
            wav, sr = torchaudio.load(basepath[0]+".wav")
            ch, wav_len = wav.shape
            ms = wav_len / sr * 1000
            if self.wav_min_ms <= ms  and ms  <= self.wav_max_ms:
                filtered_list.append(basepath[0])
                lengths.append(wav_len // (2 * self.hop_length))
            else:
                print(f"EXCEEDED LENGTH : {basepath[0]}")
        self.basepath_list = filtered_list
        self.lengths = lengths
        
    def get_ph_vocab_size(self):
        return len(self.ph_symbol_to_id) + 1 # mask用

    def get_ph_ID(self, ph_list):
        sequence = []
        for symbol in ph_list:
            symbol_id = self.ph_symbol_to_id[symbol]
            sequence += [symbol_id]
        return torch.tensor(sequence, dtype=torch.int64)

    def get_word_ID(self, word_list):
        sequence = []
        for symbol in word_list:
            symbol_id = self.word_symbol_to_id[symbol]
            sequence += [symbol_id]
        return torch.tensor(sequence, dtype=torch.int64)

    def expand_note_info(self, ph_IDs, noteID, note_dur, n_ph_pooling):
        ph_IDs_lengths = torch.tensor(ph_IDs.size(1), dtype=torch.int64)
        ph_IDs_mask = torch.unsqueeze(commons.sequence_mask(ph_IDs_lengths.view(1), ph_IDs.size(1)), 1).to(ph_IDs.dtype) # [B, 1, ph_len]
        noteID_lengths = torch.tensor(noteID.size(1), dtype=torch.int64)
        noteID_mask = torch.unsqueeze(commons.sequence_mask(noteID_lengths.view(1), noteID.size(1)), 1).to(noteID.dtype) # [B, 1, ph_len]

        attn_mask     = torch.unsqueeze(noteID_mask, 2) * torch.unsqueeze(ph_IDs_mask, -1)    # attn_mask = [B, 1, ph_len, note(word)_len]
        attn          = commons.generate_path(duration=torch.unsqueeze(n_ph_pooling,dim=1), mask=attn_mask)
        attn          = torch.squeeze(attn, dim=1).permute(0,2,1).float()                             # attn=[Batch, note_len,] 
        # expand
        noteID        = torch.matmul(noteID.float().unsqueeze(1), attn)                                            # to [Batch, inner_channel, ph_len] 
        note_dur      = torch.matmul(note_dur.float().unsqueeze(1), attn)                     # to [Batch, inner_channel, ph_len] 

        return noteID.view(-1), note_dur.view(-1)

    def get_audio(self, filename):
        #print(filename)
        audio_norm, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", "_spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            if spec.size(1) == -1:
                print("ERROR SPEC")
            torch.save(spec, spec_filename)
        return spec, audio_norm
    
    def get_dur_frame_from_e_ms(self, e_ms):
        e_ms = torch.tensor(e_ms,dtype=torch.float32)
        #frames = torch.ceil(  (e_ms/1000)*self.sampling_rate / self.hop_length )   # 切り上げ
        frames = torch.floor(  (e_ms / 1000)*self.sampling_rate / self.hop_length )   # 切り捨て
        frames = torch.diff(frames, dim=0, prepend=frames.new_zeros(1))
        #for idx in reversed(range(len(frames))):
        #    if idx == 0:
        #        continue
        #    frames[idx] = frames[idx] - frames[idx-1]
        return torch.tensor(frames, dtype=torch.int64)

    def get_ph_pooling_dur(self, ph_e, word_e):
        out = list()
        z_t = 0
        word_idx = 0
        for idx, e in enumerate(ph_e):
            idx += 1
            if word_e[word_idx] == e:
                out.append(int(idx - z_t))
                z_t = idx
                word_idx += 1
        return out

    def get_item(self, basepath):

        # labのデータは推論時存在しない
        f0          = torch.load(basepath + "_f0.pt"          ) 
        ph          = torch.load(basepath + "_ph.pt"          ) # ust or lab
        ph_e_ms     = torch.load(basepath + "_ph_e_ms.pt"     )     # lab
        #word        = torch.load(basepath + "_word.pt"        ) # ust
        #word_dur_ms = torch.load(basepath + "_word_dur_ms.pt" )     # lab
        word_e_ms   = torch.load(basepath + "_word_e_ms.pt"   )     # lab
        noteID      = torch.load(basepath + "_noteID.pt"      ) # ust
        notedur     = torch.load(basepath + "_notedur.pt"     ) # ust   
        speakerID   = 0 # 未実装 

        # tokenize and get duration
        ph_IDs              = self.get_ph_ID(ph_list=ph)
        #word_IDs            = self.get_word_ID(word_list=word)
        ph_frame_dur        = self.get_dur_frame_from_e_ms(e_ms=ph_e_ms)
        n_ph_pooling        = self.get_ph_pooling_dur(ph_e=ph_e_ms, word_e=word_e_ms)
        f0_len            = len(f0)
        notes = []
        for id in noteID:
            notes += [self.id_to_hz[int(id)]]
        noteID, notedur     = self.expand_note_info(ph_IDs=ph_IDs.view(1,-1), 
                                                    noteID      =torch.tensor(notes, dtype=torch.float32).view(1,-1),
                                                    note_dur    =torch.tensor(notedur,dtype=torch.int64).view(1,-1),
                                                    n_ph_pooling=torch.tensor(n_ph_pooling, dtype=torch.int64).view(1,-1))
        
        # padの影響で1ずれる。無問題と妄想
        if f0_len != int(torch.sum(ph_frame_dur)):
            ph_frame_dur[-1] += 1 
            
        # 保障
        assert sum(n_ph_pooling) == len(ph_IDs)
        #assert spec_len == len(vuv)
        assert len(ph_IDs) == len(noteID) 
        #assert len(word_IDs) == len(word_dur_ms)
        #assert len(word_IDs) == len(word_e_ms)

        return (torch.tensor(f0          ,            dtype=torch.float32),          
                #torch.tensor(vuv,                       dtype=torch.int64)+1,       # maskを0とする。
                torch.tensor(ph_IDs,                    dtype=torch.int64)+1,        # maskを0とする。
                torch.tensor(ph_frame_dur,              dtype=torch.int64),   
                #torch.tensor(word_dur_ms,               dtype=torch.float32) / 1000, # ここで秒になる
                torch.tensor(noteID,                    dtype=torch.float32) ,       # maskを0とする。z
                torch.tensor(speakerID,                 dtype=torch.int64)  )

    def __getitem__(self, index):
        return self.get_item(self.basepath_list[index])

    def __len__(self):
        return len(self.basepath_list)


class TextAudioCollate_SVS_f0diff():
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort( torch.LongTensor( [x[0].size(0) for x in batch] ),dim=0, descending=True )

        max_f0_len              = max([len(x[0]) for x in batch])
        #max_vuv_len            = max([len(x[1]) for x in batch])
        max_ph_IDs_len          = max([len(x[1]) for x in batch])
        max_ph_frame_dur_len    = max([len(x[2]) for x in batch])
        max_noteID_len          = max([len(x[3]) for x in batch])

        f0_lengths              = torch.LongTensor(len(batch))
        #vuv_lengths             = torch.LongTensor(len(batch))
        ph_IDs_lengths          = torch.LongTensor(len(batch))
        ph_frame_dur_lengths    = torch.LongTensor(len(batch))
        noteID_lengths          = torch.LongTensor(len(batch))
        spkids                  = torch.LongTensor(len(batch))

        f0_padded               = torch.FloatTensor(len(batch), max_f0_len)
        #vuv_padded              = torch.LongTensor(len(batch),  max_vuv_len) 
        ph_IDs_padded           = torch.LongTensor(len(batch),   max_ph_IDs_len)
        ph_frame_dur_padded     = torch.LongTensor(len(batch),   max_ph_frame_dur_len)
        noteID_padded           = torch.FloatTensor(len(batch), max_noteID_len)

        f0_padded.zero_()
        ph_IDs_padded.zero_()
        ph_frame_dur_padded.zero_()
        noteID_padded.zero_()
        spkids.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            
            f0 = row[0]
            f0_padded[i, :f0.size(0)] = f0
            f0_lengths[i] = f0.size(0)

            #vuv = row[1]
            #vuv_padded[i, :vuv.size(0)] = vuv
            #vuv_lengths[i] = vuv.size(0)
            
            ph_IDs = row[1]
            ph_IDs_padded[i, :ph_IDs.size(0)] = ph_IDs
            ph_IDs_lengths[i] = ph_IDs.size(0) 
            
            ph_frame_dur = row[2]
            ph_frame_dur_padded[i,     :ph_frame_dur.size(0)] = ph_frame_dur
            ph_frame_dur_lengths[i] =   ph_frame_dur.size(0)
            
            noteID = row[3]
            noteID_padded[i, :noteID.size(0)] = noteID
            noteID_lengths[i] = noteID.size(0)

            spkids[i] = row[4]
            

        # 次元調整
        f0_padded = torch.unsqueeze(f0_padded, dim=1)  
           
        if self.return_ids:
            return  f0_padded,              f0_lengths,             \
                    ph_IDs_padded,          ph_IDs_lengths,         \
                    ph_frame_dur_padded,                            \
                    noteID_padded,          noteID_lengths,         \
                    spkids,                                   \
                    ids_sorted_decreasing    
        
        return  f0_padded,              f0_lengths,             \
                ph_IDs_padded,          ph_IDs_lengths,         \
                ph_frame_dur_padded,                            \
                noteID_padded,          noteID_lengths,         \
                spkids

# データセット読み込み君本体
class TextAudioLoader_ALL(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams):

        self.basepath_list = load_filepaths_and_text(audiopaths_and_text, split="|")
        
        self.sampling_rate  = hparams["sampling_rate"]
        self.hop_length     = hparams["hop_length"] 
        self.filter_length  = hparams["filter_length"]
        self.win_length     = hparams["win_length"]
        self.wav_max_ms     = hparams["wav_max_ms"]
        self.wav_min_ms     = hparams["wav_min_ms"]
        self.f0_max         = hparams["f0_max"]

        self.df_f0_type = hparams["SiFiGAN_utils"]["df_f0_type"]
        self.dense_factors = hparams["SiFiGAN_utils"]["dense_factors"]
        self.upsample_scales= hparams["SiFiGANGenerator"]["upsample_scales"]

        # for SiFiGAN
        self.signal_generator = SignalGenerator(sample_rate=hparams["sampling_rate"],
                                                hop_size=hparams["hop_length"] ,
                                                sine_amp=hparams["SiFiGAN_utils"]["sine_amp"],
                                                noise_amp=hparams["SiFiGAN_utils"]["noise_amp"],
                                                signal_types=hparams["SiFiGAN_utils"]["signal_types"],)
        
        self.oto2lab, self.ph_symbol_to_id,   self.id_to_ph_symbol, \
                      self.word_symbol_to_id,self.id_to_word_symbol = get_g2p_dict_from_tabledata(table_path=hparams["oto2lab_path"], include_converter=True)
        random.seed(hparams["seed"])
        random.shuffle(self.basepath_list) # ここでシャッフルしている
        self._filter()

    # 最小以下及び最大以上のデータを弾くフィルター関数
    def _filter(self):
        filtered_list = []
        lengths = []
        for basepath in tqdm(self.basepath_list, desc="Dataset Filtering..."):
            wav, sr = torchaudio.load(basepath[0]+".wav")
            ch, wav_len = wav.shape
            ms = wav_len / sr * 1000
            if self.wav_min_ms <= ms  and ms  <= self.wav_max_ms:
                filtered_list.append(basepath[0])
                lengths.append(wav_len // (2 * self.hop_length))
            else:
                print(f"EXCEEDED LENGTH : {basepath[0]}")
        self.basepath_list = filtered_list
        self.lengths = lengths
        
    def get_ph_vocab_size(self):
        return len(self.ph_symbol_to_id) + 1 # mask用

    def get_ph_ID(self, ph_list):
        sequence = []
        for symbol in ph_list:
            symbol_id = self.ph_symbol_to_id[symbol]
            sequence += [symbol_id]
        return torch.tensor(sequence, dtype=torch.int64)
    
    def get_word_ID(self, word_list):
        sequence = []
        for symbol in word_list:
            symbol_id = self.word_symbol_to_id[symbol]
            sequence += [symbol_id]
        return torch.tensor(sequence, dtype=torch.int64)
    
    def expand_note_info(self, ph_IDs, noteID, note_dur, n_ph_pooling):
        ph_IDs_lengths = torch.tensor(ph_IDs.size(1), dtype=torch.int64)
        ph_IDs_mask = torch.unsqueeze(commons.sequence_mask(ph_IDs_lengths.view(1), ph_IDs.size(1)), 1).to(ph_IDs.dtype) # [B, 1, ph_len]
        noteID_lengths = torch.tensor(noteID.size(1), dtype=torch.int64)
        noteID_mask = torch.unsqueeze(commons.sequence_mask(noteID_lengths.view(1), noteID.size(1)), 1).to(noteID.dtype) # [B, 1, ph_len]

        attn_mask     = torch.unsqueeze(noteID_mask, 2) * torch.unsqueeze(ph_IDs_mask, -1)    # attn_mask = [B, 1, ph_len, note(word)_len]
        attn          = commons.generate_path(duration=torch.unsqueeze(n_ph_pooling,dim=1), mask=attn_mask)
        attn          = torch.squeeze(attn, dim=1).permute(0,2,1).float()                             # attn=[Batch, note_len,] 
        # expand
        noteID        = torch.matmul(noteID.float().unsqueeze(1), attn)                                            # to [Batch, inner_channel, ph_len] 
        note_dur      = torch.matmul(note_dur.float().unsqueeze(1), attn)                     # to [Batch, inner_channel, ph_len] 

        return noteID.view(-1), note_dur.view(-1)

    def get_audio(self, filename):
        #print(filename)
        audio_norm, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", "_spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            if spec.size(1) == -1:
                print("ERROR SPEC")
            torch.save(spec, spec_filename)
        return spec, audio_norm
    
    def get_dur_frame_from_e_ms(self, e_ms):
        e_ms = torch.tensor(e_ms,dtype=torch.float32)
        #frames = torch.ceil(  (e_ms/1000)*self.sampling_rate / self.hop_length )   # 切り上げ
        frames = torch.floor(  (e_ms / 1000)*self.sampling_rate / self.hop_length )   # 切り捨て
        frames = torch.diff(frames, dim=0, prepend=frames.new_zeros(1))
        #for idx in reversed(range(len(frames))):
        #    if idx == 0:
        #        continue
        #    frames[idx] = frames[idx] - frames[idx-1]
        return torch.tensor(frames, dtype=torch.int64)
    
    def get_linspace_ph_id(self, ph_idx_in_a_word):
        for idx, n_ph in enumerate(ph_idx_in_a_word):
            if idx == 0:
                data = torch.linspace(start=int(n_ph-1), end=0, steps=int(n_ph), dtype=torch.int64)
            else:
                data = torch.concat([data, torch.linspace(start=int(n_ph-1), end=0, steps=int(n_ph), dtype=torch.int64)], dim=0)
        return data
    
    def get_frame_level_word_id(self, wordID, word_dur):
        #for idx, (id, dur) in enumerate(zip(wordID, word_dur)):
        #    if idx == 0:
        #        data = torch.ones(size=(dur,),dtype=torch.int64) * id
        #    else:
        #        data = torch.concat([data, torch.ones(size=(dur,),dtype=torch.int64) * id], dim=0)
        for idx, (id, dur) in enumerate(zip(wordID, word_dur)):
            if idx == 0:
                data = torch.ones(size=(dur,),dtype=torch.int64) * idx
            else:
                data = torch.concat([data, torch.ones(size=(dur,),dtype=torch.int64) * idx], dim=0)
        return data

    def get_continuos_f0(self, f0):
        """Convert F0 to continuous F0
        Args:
            f0 (ndarray): original f0 sequence with the shape (T)
        Return:
            (ndarray): continuous f0 with the shape (T)
        """
        # get uv information as binary
        uv = np.float32(f0 != 0)
        # get start and end of f0
        if (f0 == 0).all():
            logger.warn("all of the f0 values are 0.")
            return uv, f0, False
        start_f0 = f0[f0 != 0][0]
        end_f0 = f0[f0 != 0][-1]
        # padding start and end of f0 sequence
        cf0 = copy.deepcopy(f0)
        start_idx = np.where(cf0 == start_f0)[0][0]
        end_idx = np.where(cf0 == end_f0)[0][-1]
        cf0[:start_idx] = start_f0
        cf0[end_idx:] = end_f0
        # get non-zero frame index
        nz_frames = np.where(cf0 != 0)[0]
        # perform linear interpolation
        f = interp1d(nz_frames, cf0[nz_frames])
        cf0 = f(np.arange(0, cf0.shape[0]))

        return uv, cf0, True

    def get_ph_pooling_dur(self, ph_e, word_e):
        out = list()
        z_t = 0
        word_idx = 0
        for idx, e in enumerate(ph_e):
            idx += 1
            if word_e[word_idx] == e:
                out.append(int(idx - z_t))
                z_t = idx
                word_idx += 1
        return out

    def get_item(self, basepath):

        # labのデータは推論時存在しない
        spec, wav   = self.get_audio(filename= basepath+".wav") 
        f0          = torch.load(basepath + "_f0.pt"          ) 
        vuv         = torch.load(basepath + "_vuv.pt"         ) 

        ph          = torch.load(basepath + "_ph.pt"          ) # ust or lab
        ph_dur_ms   = torch.load(basepath + "_ph_dur_ms.pt"   )     # lab
        ph_s_ms     = torch.load(basepath + "_ph_s_ms.pt"     )     # lab
        ph_e_ms     = torch.load(basepath + "_ph_e_ms.pt"     )     # lab

        word        = torch.load(basepath + "_word.pt"        ) # ust
        word_dur_ms = torch.load(basepath + "_word_dur_ms.pt" )     # lab
        word_e_ms   = torch.load(basepath + "_word_e_ms.pt"   )     # lab
        word_s_ms   = torch.load(basepath + "_word_s_ms.pt"   )     # lab

        noteID      = torch.load(basepath + "_noteID.pt"      ) # ust
        notename    = torch.load(basepath + "_notename.pt"    ) # ust    
        notedur     = torch.load(basepath + "_notedur.pt"     ) # ust    
        
        ph_idx_in_a_word = torch.load(basepath + "_n_ph_in_a_word.pt" )   # lab

        # tokenize and get duration
        ph_IDs              = self.get_ph_ID(ph_list=ph)
        word_IDs            = self.get_word_ID(word_list=word)
        ph_frame_dur        = self.get_dur_frame_from_e_ms(e_ms=ph_e_ms)
        word_frame_dur      = self.get_dur_frame_from_e_ms(e_ms=word_e_ms)
        frame_level_word_id = self.get_frame_level_word_id(wordID=word_IDs, word_dur=word_frame_dur)
        note_frame_dur      = self.get_dur_frame_from_e_ms(e_ms=torch.cumsum(torch.tensor(notedur,dtype=torch.float32),dim=0))
        ph_idx_in_a_word    = self.get_linspace_ph_id(ph_idx_in_a_word=ph_idx_in_a_word)
        n_ph_pooling        = self.get_ph_pooling_dur(ph_e=ph_e_ms, word_e=word_e_ms)
        _, spec_len         = spec.shape
        noteID, notedur     = self.expand_note_info(ph_IDs=ph_IDs.view(1,-1), 
                                                    noteID      =torch.tensor(noteID,dtype=torch.int64).view(1,-1),
                                                    note_dur    =torch.tensor(notedur,dtype=torch.int64).view(1,-1),
                                                    n_ph_pooling=torch.tensor(n_ph_pooling, dtype=torch.int64).view(1,-1))
        
        # paddingの影響で、1長くなることがあるので、その対策
        if  spec_len != len(vuv) or  spec_len != len(f0):
            #back = f0 # for debug
            f0 = f0[:spec_len]
            vuv = vuv[:spec_len]
        if spec_len != int(torch.sum(word_frame_dur)):
            word_frame_dur[-1] += 1 
        if spec_len != int(torch.sum(ph_frame_dur)):
            ph_frame_dur[-1] += 1 
        if spec_len != len(frame_level_word_id):
            frame_level_word_id = torch.tensor(frame_level_word_id)
            frame_level_word_id = torch.concat( [frame_level_word_id, torch.unsqueeze(frame_level_word_id[-1],dim=0)],  dim=0 )

        _, c_f0, _ = self.get_continuos_f0(f0) # for SiFiGAN
        Sinewave = self.signal_generator(torch.tensor(c_f0, dtype=torch.float32).view(1,1,-1))# for SiFiGAN

        prod_upsample_scales = np.cumprod(self.upsample_scales)
        df_sample_rates = [self.sampling_rate / self.hop_length * s for s in prod_upsample_scales]
        dfs = []
        for df, us in zip(self.dense_factors, prod_upsample_scales):
            dfs += [
                np.repeat(dilated_factor(c_f0, self.sampling_rate, df), us)
                if self.df_f0_type == "cf0"
                else np.repeat(dilated_factor(c_f0, self.sampling_rate, df), us)
            ]

        # 保障
        assert sum(n_ph_pooling) == len(ph_IDs)
        assert int(torch.sum(word_frame_dur)) == int(torch.sum(ph_frame_dur))
        assert len(ph_idx_in_a_word) == len(ph_IDs)
        assert spec_len == int(torch.sum(word_frame_dur))
        assert spec_len == len(frame_level_word_id)
        assert spec_len == len(f0)
        assert spec_len == len(vuv)
        assert len(ph_IDs) == len(noteID) # WORD ID が多い。おそらく"っ"の影響
        #assert len(word_IDs) == len(noteID) # WORD ID が多い。おそらく"っ"の影響
        assert len(word_IDs) == len(notename)
        assert len(word_IDs) == len(word_dur_ms)
        assert len(word_IDs) == len(word_e_ms)
        assert len(word_IDs) == len(word_s_ms)
        for i in range(len(self.dense_factors)):
            assert Sinewave.size(2) * df_sample_rates[i] == len(dfs[i]) * self.sampling_rate

        return (torch.tensor(wav,                       dtype=torch.float32),    
                torch.tensor(spec,                      dtype=torch.float32),    
                torch.tensor(f0,                        dtype=torch.float32),  
                torch.tensor(vuv,                       dtype=torch.int64)+1,    # maskを0とする。

                torch.tensor(ph_IDs,                    dtype=torch.int64)+1,    # maskを0とする。
                torch.tensor(ph_dur_ms,                 dtype=torch.float32),   
                torch.tensor(ph_frame_dur,              dtype=torch.int64),  
                torch.tensor(ph_s_ms,                   dtype=torch.float32),  
                torch.tensor(ph_e_ms,                   dtype=torch.float32),   
            
                torch.tensor(word_IDs,                  dtype=torch.int64)+1,   # maskを0とする。 
                torch.tensor(word_dur_ms,               dtype=torch.float32)/1000,  
                torch.tensor(word_frame_dur,            dtype=torch.int64),    
                torch.tensor(word_s_ms,                 dtype=torch.float32),  
                torch.tensor(word_e_ms,                 dtype=torch.float32),   

                torch.tensor(noteID,                    dtype=torch.int64)+1 ,   # maskを0とする。
                #torch.tensor(note_frame_dur,            dtype=torch.int64),     
                torch.tensor(notedur,                   dtype=torch.float32) / 1000,     
                torch.tensor(ph_idx_in_a_word,            dtype=torch.int64)+1,     # maskを0とする。
                torch.tensor(frame_level_word_id,       dtype=torch.int64)+1,    # maskを0とする。
                torch.tensor(n_ph_pooling,              dtype=torch.int64),    # maskを0とする。
                torch.tensor(c_f0,                      dtype=torch.float32),
                #torch.tensor(Sinewave,                  dtype=torch.float32),
                dfs )

    def __getitem__(self, index):
        return self.get_item(self.basepath_list[index])

    def __len__(self):
        return len(self.basepath_list)


class TextAudioCollate_ALL():
    def __init__(self, hparams, return_ids=False):
        self.return_ids = return_ids
        self.dense_factors = hparams["SiFiGAN_utils"]["dense_factors"]
        self.upsample_scales = hparams["SiFiGANGenerator"]["upsample_scales"]
        
        # for SiFiGAN
        self.signal_generator = SignalGenerator(sample_rate=hparams["sampling_rate"],
                                                hop_size=hparams["hop_length"] ,
                                                sine_amp=hparams["SiFiGAN_utils"]["sine_amp"],
                                                noise_amp=hparams["SiFiGAN_utils"]["noise_amp"],
                                                signal_types=hparams["SiFiGAN_utils"]["signal_types"],)

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort( torch.LongTensor( [x[1].size(1) for x in batch] ),dim=0, descending=True )

        max_wav_len     = max([    x[0].size(1) for x in batch])
        max_spec_len    = max([    x[1].size(1) for x in batch])
        max_f0_len      = max([len(x[2]) for x in batch])
        max_vuv_len     = max([len(x[3]) for x in batch])

        max_ph_IDs_len          = max([len(x[4]) for x in batch])
        max_ph_dur_ms_len       = max([len(x[5]) for x in batch])
        max_ph_frame_dur_len    = max([len(x[6]) for x in batch])
        max_ph_s_ms_len         = max([len(x[7]) for x in batch])
        max_ph_e_ms_len         = max([len(x[8]) for x in batch])

        max_word_IDs_len        = max([len(x[9]) for x in batch])
        max_word_dur_ms_len     = max([len(x[10]) for x in batch])
        max_word_frame_dur_len  = max([len(x[11]) for x in batch])
        max_word_s_ms_len       = max([len(x[12]) for x in batch])
        max_word_e_ms_len       = max([len(x[13]) for x in batch])

        max_noteID_len          = max([len(x[14]) for x in batch])
        max_note_frame_dur_len  = max([len(x[15]) for x in batch])
        
        max_ph_idx_in_a_word_len  = max([len(x[16]) for x in batch])
        max_frame_level_word_id_len  = max([len(x[17]) for x in batch])
        max_n_ph_pooling_len  = max([len(x[18]) for x in batch])
        
        max_c_f0_len  = max([len(x[19]) for x in batch])

        #max_Sinewave_len  = max([len(x[20]) for x in batch])

        wav_lengths     = torch.LongTensor(len(batch))
        spec_lengths    = torch.LongTensor(len(batch))
        f0_lengths      = torch.LongTensor(len(batch))
        vuv_lengths     = torch.LongTensor(len(batch))
        
        ph_IDs_lengths          = torch.LongTensor(len(batch))
        ph_dur_ms_lengths       = torch.LongTensor(len(batch))
        ph_frame_dur_lengths    = torch.LongTensor(len(batch))
        ph_s_ms_lengths         = torch.LongTensor(len(batch))
        ph_e_ms_lengths         = torch.LongTensor(len(batch))

        word_IDs_lengths            = torch.LongTensor(len(batch))
        word_dur_ms_lengths         = torch.LongTensor(len(batch))
        word_frame_dur_lengths      = torch.LongTensor(len(batch))
        word_s_ms_lengths           = torch.LongTensor(len(batch))
        word_e_ms_lengths           = torch.LongTensor(len(batch))

        noteID_lengths          = torch.LongTensor(len(batch))
        note_frame_dur_lengths  = torch.LongTensor(len(batch))

        ph_idx_in_a_word_lengths      = torch.LongTensor(len(batch))
        frame_level_word_id_lengths = torch.LongTensor(len(batch))
        n_ph_pooling_lengths        = torch.LongTensor(len(batch))
        
        c_f0_lengths        = torch.LongTensor(len(batch))
        #Sinewave_lengths        = torch.LongTensor(len(batch))

        wav_padded  = torch.FloatTensor(len(batch), 1, max_wav_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        f0_padded   = torch.FloatTensor(len(batch), max_f0_len)
        vuv_padded  = torch.LongTensor(len(batch),  max_vuv_len) 

        ph_IDs_padded       = torch.LongTensor(len(batch),   max_ph_IDs_len)
        ph_dur_ms_padded    = torch.FloatTensor(len(batch),  max_ph_dur_ms_len)
        ph_frame_dur_padded = torch.LongTensor(len(batch),   max_ph_frame_dur_len)
        ph_s_ms_padded      = torch.FloatTensor(len(batch),  max_ph_s_ms_len)
        ph_e_ms_padded      = torch.FloatTensor(len(batch),  max_ph_e_ms_len)
        
        word_IDs_padded         = torch.LongTensor(len(batch),  max_word_IDs_len)
        word_dur_ms_padded      = torch.FloatTensor(len(batch), max_word_dur_ms_len)
        word_frame_dur_padded   = torch.LongTensor(len(batch),  max_word_frame_dur_len)
        word_s_ms_padded        = torch.FloatTensor(len(batch), max_word_s_ms_len)
        word_e_ms_padded        = torch.FloatTensor(len(batch), max_word_e_ms_len)

        noteID_padded           = torch.LongTensor(len(batch), max_noteID_len)
        note_frame_dur_padded   = torch.FloatTensor(len(batch), max_note_frame_dur_len)

        ph_idx_in_a_word_padded        = torch.LongTensor(len(batch), max_ph_idx_in_a_word_len)
        frame_level_word_id_padded   = torch.LongTensor(len(batch), max_frame_level_word_id_len)
        n_ph_pooling_padded          = torch.LongTensor(len(batch), max_n_ph_pooling_len)
        
        c_f0_padded   = torch.FloatTensor(len(batch), max_c_f0_len)
        #Sinewave_padded   = torch.FloatTensor(len(batch), max_Sinewave_len)

        wav_padded.zero_()
        spec_padded.zero_()
        f0_padded.zero_()
        vuv_padded.zero_()

        ph_IDs_padded.zero_()
        ph_dur_ms_padded.zero_()
        ph_frame_dur_padded.zero_()
        ph_s_ms_padded.zero_()
        ph_e_ms_padded.zero_()

        word_IDs_padded.zero_()
        word_dur_ms_padded.zero_()
        word_frame_dur_padded.zero_()
        word_s_ms_padded.zero_()
        word_e_ms_padded.zero_()
        
        noteID_padded.zero_()
        note_frame_dur_padded.zero_()
        
        ph_idx_in_a_word_padded.zero_()
        frame_level_word_id_padded.zero_()
        n_ph_pooling_padded.zero_()
        
        c_f0_padded   .zero_()
        #Sinewave_padded   .zero_()

        dfs_padded = []
        dfs_lengths = torch.LongTensor(len(batch), len(self.upsample_scales))
        lengths=max_c_f0_len
        for idx, scales in enumerate(self.upsample_scales):
            lengths *= scales
            dfs_padded.append(torch.FloatTensor(len(batch), 1, int(lengths)).zero_())

        #dfs_batch_np = [[] for _ in range(len(self.dense_factors))]
        #dfs_batch_tensor = [[] for _ in range(len(self.dense_factors))]
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            wav = row[0]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)
            
            f0 = row[2]
            f0_padded[i, :f0.size(0)] = f0
            f0_lengths[i] = f0.size(0)

            vuv = row[3]
            vuv_padded[i, :vuv.size(0)] = vuv
            vuv_lengths[i] = vuv.size(0)
            
            ph_IDs = row[4]
            ph_IDs_padded[i, :ph_IDs.size(0)] = ph_IDs
            ph_IDs_lengths[i] = ph_IDs.size(0) 
            
            ph_dur_ms = row[5]
            ph_dur_ms_padded[i,     :ph_dur_ms.size(0)] = ph_dur_ms
            ph_dur_ms_lengths[i] =   ph_dur_ms.size(0)

            ph_frame_dur = row[6]
            ph_frame_dur_padded[i,     :ph_frame_dur.size(0)] = ph_frame_dur
            ph_frame_dur_lengths[i] =   ph_frame_dur.size(0)
            
            ph_s_ms = row[7]
            ph_s_ms_padded[i,     :ph_s_ms.size(0)] = ph_s_ms
            ph_s_ms_lengths[i] =   ph_s_ms.size(0)
            
            ph_e_ms = row[8]
            ph_e_ms_padded[i,     :ph_e_ms.size(0)] = ph_e_ms
            ph_e_ms_lengths[i] =   ph_e_ms.size(0)
            

            word_IDs = row[9]
            word_IDs_padded[i,  :word_IDs.size(0)] = word_IDs
            word_IDs_lengths[i] = word_IDs.size(0) 

            word_dur_ms = row[10]
            word_dur_ms_padded[i,     :word_dur_ms.size(0)] = word_dur_ms
            word_dur_ms_lengths[i] =   word_dur_ms.size(0)
            
            word_frame_dur = row[11]
            word_frame_dur_padded[i,     :word_frame_dur.size(0)] = word_frame_dur
            word_frame_dur_lengths[i] =   word_frame_dur.size(0)

            word_s_ms = row[12]
            word_s_ms_padded[i,     :word_s_ms.size(0)] = word_s_ms
            word_s_ms_lengths[i] =   word_s_ms.size(0)    

            word_e_ms = row[13]
            word_e_ms_padded[i,     :word_e_ms.size(0)] = word_e_ms
            word_e_ms_lengths[i] =   word_e_ms.size(0)
            
            noteID = row[14]
            noteID_padded[i, :noteID.size(0)] = noteID
            noteID_lengths[i] = noteID.size(0)

            note_frame_dur = row[15]
            note_frame_dur_padded[i, :note_frame_dur.size(0)] = note_frame_dur
            note_frame_dur_lengths[i] = note_frame_dur.size(0)

            ph_idx_in_a_word = row[16]
            ph_idx_in_a_word_padded[i, :ph_idx_in_a_word.size(0)] = ph_idx_in_a_word
            ph_idx_in_a_word_lengths[i] = ph_idx_in_a_word.size(0)

            frame_level_word_id = row[17]
            frame_level_word_id_padded[i, :frame_level_word_id.size(0)] = frame_level_word_id
            frame_level_word_id_lengths[i] = frame_level_word_id.size(0)
        
            n_ph_pooling = row[18]
            n_ph_pooling_padded[i, :n_ph_pooling.size(0)] = n_ph_pooling
            n_ph_pooling_lengths[i] = n_ph_pooling.size(0)

            c_f0 = row[19]
            c_f0_padded[i, :c_f0.size(0)] = c_f0
            c_f0_lengths[i] = c_f0.size(0)

            dfs = row[20]
            for idx in range(len(self.upsample_scales)):
                df = dfs[idx]
                dfs_padded[idx][i, 0, :len(df)] = torch.tensor(df, dtype=torch.float32)
                dfs_lengths[i, idx] = len(df)

            """
            for i in range(len(self.dense_factors)):
                dfs_batch_np[i] += [
                    dfs[i].astype(np.float32).reshape(-1, 1)
                ]  # [(T', 1), ...]

        for i in range(len(self.dense_factors)):
            dfs_batch_tensor[i] = torch.FloatTensor(np.array(dfs_batch_np[i])).transpose(
                    2, 1
                )  # (B, 1, T')
                
            """
        # 次元調整
        f0_padded = torch.unsqueeze(f0_padded, dim=1)  
            
        # バッチ単位で作成
        Sinewaves = self.signal_generator(c_f0_padded.unsqueeze(1)).float()

        if self.return_ids:
            return  wav_padded,             wav_lengths,            \
                    spec_padded,            spec_lengths,           \
                    f0_padded,              f0_lengths,             \
                    vuv_padded,             vuv_lengths,            \
                    ph_IDs_padded,          ph_IDs_lengths,         \
                    ph_dur_ms_padded,       ph_dur_ms_lengths,      \
                    ph_frame_dur_padded,    ph_frame_dur_lengths,   \
                    ph_s_ms_padded,         ph_s_ms_lengths,        \
                    ph_e_ms_padded,         ph_e_ms_lengths,        \
                    word_IDs_padded,        word_IDs_lengths,       \
                    word_dur_ms_padded,     word_dur_ms_lengths,    \
                    word_frame_dur_padded,  word_frame_dur_lengths, \
                    word_s_ms_padded,       word_s_ms_lengths,      \
                    word_e_ms_padded,       word_e_ms_lengths,      \
                    noteID_padded,          noteID_lengths,         \
                    note_frame_dur_padded,  note_frame_dur_lengths, \
                    ph_idx_in_a_word_padded,  ph_idx_in_a_word_lengths, \
                    frame_level_word_id_padded, frame_level_word_id_lengths, \
                    n_ph_pooling_padded, n_ph_pooling_lengths, \
                    c_f0_padded, c_f0_lengths, \
                    dfs_padded, dfs_lengths, \
                    Sinewaves, \
                    ids_sorted_decreasing    
        
        return  wav_padded,             wav_lengths,            \
                spec_padded,            spec_lengths,           \
                f0_padded,              f0_lengths,             \
                vuv_padded,             vuv_lengths,            \
                ph_IDs_padded,          ph_IDs_lengths,         \
                ph_dur_ms_padded,       ph_dur_ms_lengths,      \
                ph_frame_dur_padded,    ph_frame_dur_lengths,   \
                ph_s_ms_padded,         ph_s_ms_lengths,        \
                ph_e_ms_padded,         ph_e_ms_lengths,        \
                word_IDs_padded,        word_IDs_lengths,       \
                word_dur_ms_padded,     word_dur_ms_lengths,    \
                word_frame_dur_padded,  word_frame_dur_lengths, \
                word_s_ms_padded,       word_s_ms_lengths,      \
                word_e_ms_padded,       word_e_ms_lengths,      \
                noteID_padded,          noteID_lengths,         \
                note_frame_dur_padded,  note_frame_dur_lengths, \
                ph_idx_in_a_word_padded,  ph_idx_in_a_word_lengths, \
                frame_level_word_id_padded, frame_level_word_id_lengths, \
                n_ph_pooling_padded, n_ph_pooling_lengths, \
                c_f0_padded, c_f0_lengths, \
                dfs_padded, dfs_lengths, \
                Sinewaves


"""Multi speaker version"""

class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        return (text, spec, wav, sid)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        #audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size

if __name__ == "__main__":
    print(1 * False)

    f0_length = 678
    IDs_length = 55
    Batch = 3
    n_speaker = 10
    vocab_size= 20
    max_step = 1000

    dummy_f0        = torch.rand   (                size=( f0_length, ))
    dummy_f0_len    = torch.randint(f0_length ,     size=(1,))
    dummy_vuv       = torch.randint(1,              size=(f0_length,))
    dummy_vuv_len   = torch.randint(f0_length,      size=(1,))
    dummy_IDs       = torch.randint(vocab_size,     size=(IDs_length,))
    dummy_IDs_len   = torch.randint(IDs_length,     size=(1,))
    dummy_attn      = torch.randint(1,              size=(f0_length, IDs_length))
    dummy_g         = torch.randint(n_speaker,      size=(1,))
    dummy_diff_steps= torch.randint(max_step,       size=(1,))

    test = TextAudioCollate()
    batch = [[dummy_f0, dummy_attn, dummy_IDs, dummy_vuv] for x in range(5)]
    _ = test.__call__(batch=batch)