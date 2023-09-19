import warnings
warnings.filterwarnings(action='ignore')

import os
import time
import torch
import utils
import argparse
import utils
from models import VITS2_based_SiFiSinger, DiffusionModels
import soundcard as sc
import soundfile as sf
from singDB_loader import get_ust_info, get_g2p_dict_from_tabledata
import commons
from sifigan.utils import dilated_factor        
from sifigan.utils.features import SignalGenerator
import numpy as np
import logger
import copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def inference(args):

    ###### CPU Inference is not allowed ######
    assert torch.cuda.is_available(), "CPU Inference is not allowed."
    ##########################################

    config_path = args.config
    F0_model_path = args.F0diff_model_path
    Synthesizer_model_path = args.Synthesizer_model_path
    ust_path = args.UST_path
    
    ask_retake = args.ask_retake

    # load config.json
    hps = utils.get_hparams(config_path)
    oto2lab_tablepath = hps["oto2lab_path"]

    # save directory process
    if args.is_save is True:
        save_dir = os.path.join("./infer_logs/")
        os.makedirs(save_dir, exist_ok=True)

    # preprocess
    g2p, ph_symbol_to_id, _, _, _ = get_g2p_dict_from_tabledata(oto2lab_tablepath, include_converter=True)
    inference_loader = preprocess_dataset(get_ust_info(ust_path), g2p, ph_symbol_to_id, hps)

    # load checkpoint
    SVS = VITS2_based_SiFiSinger(
      hps = hps,
      n_vocab=len(ph_symbol_to_id) + 1, # +1 means masked symbol id
      spec_channels=hps["spec_encoder"]["spec_channels"],
      segment_size=int(hps["segments_size"] // hps["hop_length"]),
      n_speakers=hps["common"]["n_speaker"],
      gin_channels=hps["common"]["gin_channels"],
      **hps["hifi_gan"]).cuda()
    _ = SVS.eval()
    utils.load_model_diffsize(Synthesizer_model_path, SVS, hps, None)

    # モデルを作成
    F0DIFF = DiffusionModels(hps=hps, vocab_size=len(ph_symbol_to_id)+1, training=False).cuda()
    _ = F0DIFF.eval()
    utils.load_model_diffsize(F0_model_path, F0DIFF, hps, None)

    # for SiFiGAN
    df_f0_type = hps["SiFiGAN_utils"]["df_f0_type"]
    dense_factors = hps["SiFiGAN_utils"]["dense_factors"]
    upsample_scales= hps["SiFiGANGenerator"]["upsample_scales"]
    prod_upsample_scales = np.cumprod(upsample_scales)
    df_sample_rates = [hps["sampling_rate"] / hps["hop_length"] * s for s in prod_upsample_scales]
    signal_generator = SignalGenerator(sample_rate=hps["sampling_rate"],
                                            hop_size=hps["hop_length"] ,
                                            sine_amp=hps["SiFiGAN_utils"]["sine_amp"],
                                            noise_amp=hps["SiFiGAN_utils"]["noise_amp"],
                                            signal_types=hps["SiFiGAN_utils"]["signal_types"],)

    # play audio by system default
    speaker = sc.get_speaker(sc.default_speaker().name)
    
    out_voices = list()
    diff_lists = list()
    total_lengths = list()

    #import glob
    #for path in glob.glob("./infer_logs/*.wav"):
    #    y, sr = sf.read(path)
    #    out_voices.append(y)

    # required_grad is False
    with torch.inference_mode():
        for idx, (ph_seq, ph_length, word_frame_dur, word_lengths, word_dur_ms, ph_idx_in_a_word, n_ph_seq,\
             noteid_seq, noteid_length) in enumerate(inference_loader):
            speakerID = torch.zeros(size=(1,), dtype=torch.int64) # speaker id is fixed
            
            # for retake flag
            retake_flag = True
            while retake_flag:

                # measure the execution time 
                torch.cuda.synchronize()
                start = time.time()

                # 長い休符の時は、無音出力で終わらせる。
                if ph_seq[0][0] == 35 and word_dur_ms[0][0]>3:
                    length=int(word_dur_ms[0][0] * hps["sampling_rate"])
                    data = np.zeros(shape=(length,))
                    save_path = os.path.join(save_dir, str(idx).zfill(3)+".wav")
                    sf.write(
                        file=save_path,
                        data=data,
                        samplerate=hps["sampling_rate"],
                        format="WAV")
                    out_voices.append(data)
                    diff_lists.append(0)
                    total_lengths.append(length)
                    retake_flag = False
                    print(f"[INFO] Silent wavs is saved at {save_path}")
                    continue

                # duration predict and encoding
                z_spec, _, w_ceil, g, diff = SVS.encode_and_dp(ph_IDs=ph_seq.cuda(),
                                                    ph_IDs_lengths=ph_length.cuda(),
                                                    speakerID=speakerID.cuda(),
                                                    word_frame_dur=word_frame_dur.cuda(),
                                                    word_frame_dur_lenngths=word_lengths.cuda(),
                                                    word_dur_ms=word_dur_ms.cuda(),
                                                    ph_word_flag=ph_idx_in_a_word.cuda(),
                                                    n_ph_pool=n_ph_seq.cuda(),
                                                    noise_scale=1.0)

                # f0 diffusion [1,1,T']
                f0_pd = F0DIFF.sampling(condition=[ph_seq.cuda(), ph_length.cuda(), w_ceil.cuda(), 
                                      noteid_seq.cuda(), noteid_length.cuda(), None, speakerID.cuda()])

                # Below threshold, truncated
                f0_pd [f0_pd<int(hps["f0_min"])] = 0

                # calc continuousf0 and dilated factors
                _, continuous_f0, _ = get_continuos_f0(f0_pd[0][0].to('cpu').detach().numpy().copy())
                Sinewave = signal_generator(torch.tensor(continuous_f0, dtype=torch.float32).view(1,1,-1))# for SiFiGAN
                dfs = []
                for df, us in zip(dense_factors, prod_upsample_scales):
                    dfs += [
                        torch.from_numpy(np.repeat(dilated_factor(continuous_f0, hps["sampling_rate"], df), us).astype(np.float32)).clone()
                        if df_f0_type == "cf0"
                        else torch.from_numpy(np.repeat(dilated_factor(continuous_f0, hps["sampling_rate"], df), us).astype(np.float32)).clone()
                    ]

                # check length 
                for i in range(len(dense_factors)):
                    assert Sinewave.size(2) * df_sample_rates[i] == len(dfs[i]) * hps["sampling_rate"]

                # synthesize voice=[1,1,T]
                voice = SVS.synthesize(Sinewave.float().cuda(), z_spec, dfs, g)
                
                # measure the execution time 
                torch.cuda.synchronize()
                elapsed_time = time.time() - start
                print(f"Gen Time : {elapsed_time}")

                if ask_retake is True:
                    speaker.play(data=voice[0][0].to('cpu').detach().numpy().copy(),
                                 samplerate=hps["sampling_rate"],
                                 channels=1)
                    x = input("OK:0, Retake:1  select==>")
                    if int(x) == 1:
                        pass
                    else:
                        retake_flag = False

                else:
                    retake_flag = False
            
            if ph_seq[0][0] == 35 and word_dur_ms[0][0]>3:
                continue

            diff_lists.append(diff) # first ph duration difference
            total_lengths.append(int(torch.sum(word_dur_ms) * hps["sampling_rate"])) # total note length 
            out_voices.append(voice[0][0].to('cpu').detach().numpy().copy()) # synthesized voice

            if args.is_save is True:
                data = voice[0][0].to('cpu').detach().numpy().copy()
                save_path = os.path.join(save_dir, str(idx).zfill(3)+".wav")
                sf.write(
                    file=save_path,
                    data=data,
                    samplerate=hps["sampling_rate"],
                    format="WAV")
                print(f"[INFO] Save synthesized voice at {save_path}")

                f0_pd = f0_pd[0][0].to('cpu').detach().numpy().copy()
                generate_graph(vector=f0_pd , 
                            label="F0 gt",
                            color="red", 
                            x_label = 'Frames',
                            y_label = "f0",
                            savename = os.path.join(save_dir, str(idx).zfill(3)+".png"))

    concat_voices(out_voices, save_dir, hps, diff_lists, total_lengths)
    return 0

def concat_voices(out_voices,save_dir, hps, diff_lists, total_lengths):
    lengths = 0
    for idx, voice in enumerate(out_voices):
        lengths += len(voice)

    wavs = np.zeros(shape=(int(lengths,)))

    z_idx = 0
    for idx, (voice, diff, total) in enumerate(zip(out_voices, diff_lists, total_lengths)):
        start_idx = z_idx-diff
        wavs[start_idx:start_idx+len(voice)] += voice
        z_idx += total

    sf.write(file=os.path.join(save_dir, "full_chorus.wav"),
             data=wavs[:z_idx],
             samplerate=hps["sampling_rate"],
             format="WAV")

def generate_graph(vector, 
                   label="NoName",
                   color="blue", 
                   title="Title",
                   x_label = 'Frames',
                   y_label = "y_labels",
                   figsize=(20,5),
                   savename=None):
   
    fig = plt.figure(figsize=figsize) 
    x = np.arange(0, len(vector))
    plt.plot(x, vector, label=label, color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.ylim(y_val_min, y_val_max)
    fig.canvas.draw() 
    plt.savefig(savename)
    plt.clf()
    plt.close()
    
def preprocess_dataset(UST_data, g2p, ph2id, hps):

    with open(hps["noteid2hz_txt_path"], mode="r", encoding="utf-8") as f:
        lines = f.readlines()
    id_to_hz = {}
    for idx, line in enumerate(lines):
        id, hz = line.split(",")
        id_to_hz[idx-1] = float(hz)

    word_seq = list()
    for idx, ust in enumerate(UST_data):
        #print(ust)
        ms = ust["length_ms"]
        if ms == 0:
            continue
        noteid = float(id_to_hz[ust["note_MIDI"]])
        lyric = ust["lyric"]
        if "っ" in lyric and len(lyric)>1:
            txt = ""   
            word_ph = ""   
            # check cl
            for w in lyric:
                if w == "っ":
                    word_ph += g2p[txt] + " " + g2p[w] # pull from dict
                    txt = "" #reset
                else:
                    txt += w # stack
            # last pull
            if txt != "":
                word_ph += g2p[txt]
        else:      
            word_ph = g2p[lyric]
        word_ph = word_ph.split(" ")
        n_ph = len(word_ph)
        id_seq = []
        for ph in word_ph:
            #print(ph)
            id = ph2id[ph]
            id_seq += [id]
        word_seq .append( [ms, id_seq, n_ph, noteid])

    batch_process = list()
    total_time = 0
    ph_seq = []
    n_ph_seq = []
    noteid_seq = []
    ms_seq =[]
    for ms, ph_ids, n_ph, noteid in word_seq:
        ms_seq      += [ms]
        ph_seq      += [ph_ids]
        n_ph_seq    += [n_ph ]
        noteid_seq  += [noteid]

        # 初手長い休憩の時
        if ph_ids[0] == ph2id[g2p["R"]] and ms > 3000 and total_time == 0: 
            batch_process.append([ms_seq, ph_seq, n_ph_seq, noteid_seq])
            ph_seq = []
            n_ph_seq = []
            noteid_seq = []
            ms_seq =[]
            total_time = 0

        # 途中で来たR時
        if ph_ids[0] == ph2id[g2p["R"]] and total_time!=0:
            
            if ms > 3000:
                batch_process.append([ms_seq[:-1], ph_seq[:-1], n_ph_seq[:-1], noteid_seq[:-1]])
                batch_process.append([ms_seq[-1], ph_seq[-1], n_ph_seq[-1], noteid_seq[-1]])
            else:
                batch_process.append([ms_seq, ph_seq, n_ph_seq, noteid_seq])

            ph_seq = []
            n_ph_seq = []
            noteid_seq = []
            ms_seq =[]
            total_time = 0

        total_time += ms
    if ms_seq != []:
        batch_process.append([ms_seq, ph_seq, n_ph_seq, noteid_seq])

    inference_loader = []
    for ms_seq, ph_seq, n_ph_seq, noteid_seq in batch_process:
        ph_seq_t = []
        for p in ph_seq:
            if type(p) == int:
                ph_seq_t.extend([p])
            else:
                ph_seq_t.extend(p)

        word_dur_ms     = torch.tensor(ms_seq,  dtype=torch.float32).view(-1)
        ph_seq          = torch.tensor(ph_seq_t,  dtype=torch.int64).view(-1)
        ph_length       = torch.tensor(len(ph_seq),  dtype=torch.int64).view(-1)
        n_ph_seq        = torch.tensor(n_ph_seq,  dtype=torch.int64).view(-1)
        noteid_seq      = torch.tensor(noteid_seq,  dtype=torch.float32).view(-1)

        e_ms=torch.cumsum(torch.tensor(word_dur_ms,dtype=torch.float32),dim=0).view(-1)
        if e_ms.size(0) == 1:
            word_frame_dur = e_ms
        else:
            word_frame_dur = get_dur_frame_from_e_ms(e_ms, sr=hps["sampling_rate"], hop_len=hps["hop_length"])
        word_lengths       = torch.tensor(word_frame_dur.size(0), dtype=torch.int64).view(-1)
        ph_idx_in_a_word   = get_linspace_ph_id(n_ph_seq)
        noteid_seq         = expand_note_info(ph_seq.unsqueeze(0), noteid_seq.unsqueeze(0), n_ph_seq.unsqueeze(0))
        noteid_length      = torch.tensor(len(noteid_seq), dtype=torch.int64)
        inference_loader.append([ph_seq.view(1,-1)+1 , 
                                 ph_length.view(-1), 
                                 word_frame_dur.view(1,-1), 
                                 word_lengths.view(-1), 
                                 word_dur_ms.view(1,-1) / 1000, 
                                 ph_idx_in_a_word.view(1,-1)+1, 
                                 n_ph_seq.view(1,-1),
                                 noteid_seq.view(1,-1)+1,
                                 noteid_length.view(-1)])
    return inference_loader

def get_continuos_f0(f0):
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

def expand_note_info( ph_IDs, noteID, n_ph_pooling):
    ph_IDs_lengths = torch.tensor(ph_IDs.size(1), dtype=torch.int64)
    ph_IDs_mask = torch.unsqueeze(commons.sequence_mask(ph_IDs_lengths.view(1), ph_IDs.size(1)), 1).to(ph_IDs.dtype) # [B, 1, ph_len]
    noteID_lengths = torch.tensor(noteID.size(1), dtype=torch.int64)
    noteID_mask = torch.unsqueeze(commons.sequence_mask(noteID_lengths.view(1), noteID.size(1)), 1).to(noteID.dtype) # [B, 1, ph_len]
    attn_mask     = torch.unsqueeze(noteID_mask, 2) * torch.unsqueeze(ph_IDs_mask, -1)    # attn_mask = [B, 1, ph_len, note(word)_len]
    attn          = commons.generate_path(duration=torch.unsqueeze(n_ph_pooling,dim=1), mask=attn_mask)
    attn          = torch.squeeze(attn, dim=1).permute(0,2,1).float()                             # attn=[Batch, note_len,] 
    # expand
    noteID        = torch.matmul(noteID.float().unsqueeze(1), attn).float()                                         # to [Batch, inner_channel, ph_len] 
    return noteID.view(-1)

def get_linspace_ph_id( ph_idx_in_a_word):
    for idx, n_ph in enumerate(ph_idx_in_a_word):
        if idx == 0:
            data = torch.linspace(start=int(n_ph-1), end=0, steps=int(n_ph), dtype=torch.int64)
        else:
            data = torch.concat([data, torch.linspace(start=int(n_ph-1), end=0, steps=int(n_ph), dtype=torch.int64)], dim=0)
    return data

def get_dur_frame_from_e_ms(e_ms, sr, hop_len):
    e_ms = torch.tensor(e_ms,dtype=torch.float32)
    frames = torch.floor(  (e_ms / 1000)*sr / hop_len )   # 切り捨て
    frames = torch.diff(frames, dim=0, prepend=frames.new_zeros(1))
    return torch.tensor(frames, dtype=torch.int64)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        #required=True,
                        default="./configs/config.yaml" ,    
                        help='Path to configuration file')
    parser.add_argument('--F0diff_model_path',
                        type=str,
                        #required=True,
                        default="./inference_models/F0DIFF.pth",
                        help='Path to checkpoint')
    parser.add_argument('--Synthesizer_model_path',
                        type=str,
                        #required=True,
                        default="./inference_models/SVS.pth",
                        help='Path to checkpoint')
    parser.add_argument('--UST_path',
                        type=str,
                        #required=True,
                        default="./inference_models/ﾄﾞﾚﾐﾌｧｿﾗｼﾄﾞ.ust",
                        help='Path to checkpoint')
    parser.add_argument('--ask_retake',
                        type=str,
                        default=False,
                        help='Whether to save output or not')
    parser.add_argument('--is_save',
                        type=str,
                        default=True,
                        help='Whether to save output or not')
    args = parser.parse_args()
    
    print(f"[INFO] AI Retake function is {args.ask_retake}")
    inference(args)