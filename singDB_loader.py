import utaupy
import music21 as m21
import mido
import pyopenjtalk
import shutil
from time import sleep
import os
import torch

def get_midi_info(path):
    midi = mido.MidiFile(path)
    for tracks in midi.tracks:
        track_name = tracks.name

        # GET BPM
        lines = list()
        if track_name == "Control":
            for track in tracks:
                try:
                    if "@set tempo" in track.text:
                        _, BPM =  track.text .split("=")
                except:
                    pass 
        # GET notes and dur
        else:
            for track in tracks:
                try:
                    if float(track.time) > 0:
                        tick_dur = float(track.time)
                        try:
                            note = str(track.note)
                        except:
                            note = "rest"
                        line = {"Note" : note, "tick_dur" : tick_dur, "ms_dur":tick2ms(float(BPM), tick_dur)}
                        lines.append(line)
                except:
                    continue 
    #print(midi)
    return BPM, lines

def get_ust_info(path): 
# Note Number : https://www.asahi-net.or.jp/~hb9t-ktd/music/Japan/Research/DTM/freq_map.html
    ustobj = utaupy.ust.load(path)
    lines = list()
    for note in ustobj.notes:
        length = note.length
        length_ms = note.length_ms
        notename = note.notename
        note_MIDI = note.notenum
        lyric = note.lyric

        line = {"length"    : length,
                "length_ms" : length_ms,
                "notename"  : notename,
                "note_MIDI" : note_MIDI,
                "lyric"     : lyric}
        lines.append(line)
    
    return lines

def get_lab_info(path):
    with open(path, mode="r", encoding="utf-8") as f:
        lines_r = f.readlines()
    lines = list()
    for line_r in lines_r:
        line_r = line_r.replace("\n", "")
        start_time_100us, end_time_100us, ph = line_r.split(" ")
        t_start = float(start_time_100us)/10000
        t_end = float(end_time_100us)/10000
        line = {"start_ms":t_start,
                "end_ms": t_end,
                "dur_ms": float(t_end - t_start),
                "ph": ph}
        lines.append(line)

    return lines


def get_lab_info_no_scaling(path):
    with open(path, mode="r", encoding="utf-8") as f:
        lines_r = f.readlines()
    lines = list()
    for line_r in lines_r:
        line_r = line_r.replace("\n", "")
        start_time_100us, end_time_100us, ph = line_r.split(" ")
        t_start = int(start_time_100us)
        t_end = int(end_time_100us)
        line = {"start_ms":t_start,
                "end_ms": t_end,
                "dur_ms": int(t_end - t_start),
                "ph": ph}
        lines.append(line)

    return lines

def get_musicxml_info(filepath):
    try:
        piece = m21.converter.parse(filepath)
        lines = list()

        #音またはrestをlist_noteに、長さをlist_durationに格納する
        for n in piece.flat.notesAndRests:
            if type(n) == m21.note.Note:
                pitch = str(n.pitch)
                beats_dur = str(n.duration.quarterLength)
                word_lyric = str(n.lyric)
                sec_dur = float(n.seconds)
            elif type(n) == m21.note.Rest:
                pitch = str(n.name)
                beats_dur = str(n.duration.quarterLength)
                word_lyric = "rest"
                sec_dur = float(n.seconds)
            else:
                print(f"ERROR : {filepath}  END.")
                exit()
            line = {"pitch" : pitch, 
                    "beats_dur" : beats_dur,
                    "word_lyric" : word_lyric,
                    "ms_dur" : str(sec_dur*1000)}
            lines.append(line)

    except ZeroDivisionError:
        print('Zero Division Error')

    return lines 

def get_g2p_dict_from_training_data(word_list_path):
    word_list = torch.load(word_list_path)
    word_symbol_to_id = {s.replace("\n", ""): i for i, s in enumerate(sorted(set(word_list)))} # oto table用
    id_to_word_symbol = {i: s.replace("\n", "") for i, s in enumerate(sorted(set(word_list)))} # oto table用
    return word_symbol_to_id, id_to_word_symbol

def get_g2p_dict_from_tabledata(table_path, include_converter=False):
    with open(table_path, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
    g2p_dict = {}
    ph_list = list()
    oto_list = list()
    for line in lines:
        line = line.split(" ")
        oto = line.pop(0).replace("\n","")
        oto_list.append(oto)
        ph = " ".join(line).replace("\n","")
        ph_list.extend(line)
        g2p_dict[oto] = ph
    if include_converter is True:
        ph_symbol_to_id = {s.replace("\n", ""): i for i, s in enumerate(sorted(set(ph_list)))}
        id_to_ph_symbol = {i: s.replace("\n", "") for i, s in enumerate(sorted(set(ph_list)))}
        word_symbol_to_id = {s.replace("\n", ""): i for i, s in enumerate(sorted(set(oto_list)))} # oto table用
        id_to_word_symbol = {i: s.replace("\n", "") for i, s in enumerate(sorted(set(oto_list)))} # oto table用
        return g2p_dict, ph_symbol_to_id, id_to_ph_symbol, word_symbol_to_id, id_to_word_symbol
    else:
        return g2p_dict

def g2p_pyopenjtalk(kana):
    return pyopenjtalk.g2p(kana)

def tick2ms(BPM, tick):
    tick_units = (60/BPM) *1000 /480
    ms = tick_units * tick
    return ms

def get_annotated_data(ust_path, lab_path, ono2lab_table_path):
    lab_collate(lab_path) # silを完全消去→pauへ
    data_UST = get_ust_info(ust_path)
    data_LAB = get_lab_info(lab_path)
    g2p_dict = get_g2p_dict_from_tabledata(ono2lab_table_path)

    lab_lines = list()
    for line in data_LAB:
        lab_lines.append({"ph":line["ph"] , "ph_start_point_ms" :line["start_ms"], "ph_end_point_ms" :line["end_ms"]})

    word_lines = list()
    for line in data_UST:
        text = line["lyric"]
        # "とっ"のような単語が入ってくるが、"と"、"っ"の二つで対応するための処理
        if "っ" in text and len(text)>1:
            try:
                word_ph = g2p_dict[text] # もし辞書に有ればそのまま使う。
            except:
                txt = ""   
                word_ph = ""   
                # check cl
                for w in text:
                    if w == "っ":
                        word_ph += g2p_dict[txt] + " " + g2p_dict[w] # pull from dict
                        txt = "" #reset
                    else:
                        txt += w # stack
                # last pull
                if txt != "":
                    word_ph += g2p_dict[txt]
        else:      
            word_ph = g2p_dict[text]

        if text == "R":
            note_num = -1
            notename = "rest"
        else:
            note_num = line["note_MIDI"]
            notename = line["notename"]
        note_dur = line["length_ms"]

        # 大文字で判別
        start_point = 0 
        end_point = 0
        ph_list = list()
        for idx , WORD_PH_UNIT in enumerate(word_ph.upper().split(" ")):
            line_lab = lab_lines.pop(0) # 一個ずつ出していく
            LAB_PH = line_lab["ph"].upper()

            # 初回なら開始位置保存
            if idx == 0:
                start_point = line_lab["ph_start_point_ms"]
            
            # ワードが一致するまで終了位置を保存
            if WORD_PH_UNIT == LAB_PH:
                end_point = line_lab["ph_end_point_ms"]

            elif LAB_PH == "SIL":
                end_point = line_lab["ph_end_point_ms"]
                text = "R" 

            # 一致しなかったら強制終了
            else:
                print("ERROR ph mismatch")
                exit()
            ph_list.append([   line_lab["ph"],                  \
                               line_lab["ph_start_point_ms"],   \
                               line_lab["ph_end_point_ms"],     \
                               line_lab["ph_end_point_ms"] - line_lab["ph_start_point_ms"] \
                               ])
        word_line = {"word": text, 
                     "ph": ph_list ,
                     "start_ms":start_point, 
                     "end_ms": end_point, 
                     "dur_ms": end_point-start_point,
                     "note_num": note_num,
                     "notename": notename,
                     "note_dur": note_dur}
        word_lines.append(word_line)
    return word_lines


def lab_collate(path):

    lab_list = get_lab_info_no_scaling(path)
    temp_txt = "temp.lab"
    z_line   = lab_list.pop(0)
    z_s_time = z_line["start_ms"]
    z_e_time = z_line["end_ms"]
    z_ph     = z_line["ph"]
    with open(temp_txt, mode="a", encoding="utf-8") as f:
        f.write(str(z_s_time) + " ")
        for  line in lab_list:
            #s_time, e_time, ph = line
            s_time = line["start_ms"]
            e_time = line["end_ms"]
            ph = line["ph"]

            # z_e_timeとs_timeの書き込みは、pau or silが連続していない時のみ。
            if z_ph == "sil" or z_ph =="pau":
                if ph == "sil" or ph == "pau":
                    pass
                else:
                    f.write(str(z_e_time) + " " + z_ph + "\n" + str(s_time) + " ")
            else:
                f.write(str(z_e_time) + " " + z_ph + "\n" + str(s_time) + " ")
            z_s_time = s_time
            z_e_time = e_time
            z_ph = ph
        f.write(str(e_time) + " " + ph + "\n")
    sleep(1)
    shutil.copy(temp_txt, path)
    os.remove(temp_txt)
    


if __name__ == "__main__":
    path = "./kana2phonemes_002_oto2lab.table"
    get_g2p_dict_from_tabledata(path)
    #lab_collate("./DATABASE/1st_color/1st_color.lab")
    

