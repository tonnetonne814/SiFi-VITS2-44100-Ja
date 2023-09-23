import torch.nn as nn
from diffusion.diffusion_utils import calc_diffusion_step_embedding
#from diffusion.diffusion_multinomial import MultinomialDiffusion
from diffusion.diffusion_model import F0_Diffusion
from diffusion.diffusers_modules import Downsample1D, Upsample1D, ResBlock1D, OutConv1DBlock
from attentions import Decoder as CrossAttn
from attentions import Encoder as SelfAttn
import commons, modules, attentions, math, torch, copy, logging
import math
from commons import generate_path
from tqdm import tqdm 
import torch
import torch.nn as nn
from collections import OrderedDict
#from vdecoder.hifigan.models import Generator # from so-vits-svc
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn import Conv1d,   Conv2d
from commons import get_padding
from commons import convert_logdur_to_intdur
import copy
from sifigan.models.generator import SiFiGANGenerator
from singDB_loader import get_g2p_dict_from_tabledata
import monotonic_align


class VITS2_based_SiFiTTS(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    hps,
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    n_speakers,
    gin_channels,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = hps["common"]["n_speaker"]
    self.gin_channels = hps["common"]["gin_channels"]
    self.transformer_flow_type = "fft" # "fft" / "mono_layer" / "pre_conv" ### When mono_layer and pre_conv, kl div loss went negative.
    self.current_mas_noise_scale = float(hps["VITS2_config"]["mas_noise_scale"])
    self.enc_gin_channels = gin_channels

    # VITS2 Text Encoder
    self.enc_p = TextEncoder_VITS2(n_vocab,hps["note_encoder"]["n_note"]+1,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=self.enc_gin_channels)

    # SiFi Decoder
    self.dec = SiFiGANGenerator(**hps["SiFiGANGenerator"])
    self.upsample_scales= hps["SiFiGANGenerator"]["upsample_scales"]

    # VITS1 Encoder
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)

    # VITS2 Flow
    self.flow = ResidualCouplingTransformersBlock(
       inter_channels, 
       hidden_channels, 
       5, 
       1, 
       4, 
       gin_channels=gin_channels, 
       use_transformer_flows=True,
       transformer_flow_type=self.transformer_flow_type
       )

    # FastSpeech2 DurationPredictor
    self.dp = VariancePredictor(hps=hps, 
                                input_size      =hps["dur_predictor"]["input_size"],
                                filter_size     =hps["dur_predictor"]["filter_size"],
                                kernel          =hps["dur_predictor"]["kernel_size"],
                                conv_output_size=hps["dur_predictor"]["filter_size"],
                                dropout         =hps["dur_predictor"]["dropout"],
                                n_speaker=hps["common"]["n_speaker"])

    if n_speakers >= 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)

    self.ms_per_frame = hps["sampling_rate"] / (hps["hop_length"] * 1000)
    self.hop_length = hps["hop_length"]
    self.oto2lab, self.ph_to_id,   self.id_to_ph, _,_ = get_g2p_dict_from_tabledata(table_path=hps["oto2lab_path"], 
                                                                                    include_converter=True)
    try:
      self.ph_statistics = torch.load(hps["ph_statistics_path"]) #{ph:[mean,var]} 
      print(f"[INFO] Loaded :", hps["ph_statistics_path"])
    except:
      self.ph_statistics = False

  def forward(self,
                spec, spec_lengths,
                ph_IDs, ph_IDs_lengths,
                dfs, 
                sinewave,
                speakerID):
    
    if self.n_speakers > 0:
      g = self.emb_g(speakerID).unsqueeze(-1) # [B, hidden, 1]
    else:
      g = None

    # posterior encoder
    z_spec, z_spec_m_q, z_spec_logs_q, spec_mask    = self.enc_q(spec, spec_lengths.float(), g=g)           # z_spec=[B, hidden, spec_len]

    # Flow 
    z_spec_text = self.flow(z_spec, spec_mask, g=g) # z_spec_text=[B, hidden, spec_len]

    # prior encoder 
    H_ph,   H_ph_m_p,   H_ph_logs_p,  H_ph_mask  = self.enc_p(ph_IDs, ph_IDs_lengths, 
                                                              w_dur_ms=None,
                                                              ph_w_idx=None,
                                                              g=g)       # H_ph=[B, hidden, ph_len]

    with torch.no_grad():
      # negative cross-entropy
      s_p_sq_r = torch.exp(-2 * H_ph_logs_p) # [b, d, t]
      neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - H_ph_logs_p, [1], keepdim=True) # [b, 1, t_s]
      neg_cent2 = torch.matmul(-0.5 * (z_spec_text ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent3 = torch.matmul(z_spec_text.transpose(1, 2), (H_ph_m_p * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent4 = torch.sum(-0.5 * (H_ph_m_p ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
      neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

      epsilon = torch.std(neg_cent) * torch.randn_like(neg_cent) * self.current_mas_noise_scale
      neg_cent = neg_cent + epsilon

      attn_mask = torch.unsqueeze(H_ph_mask, 2) * torch.unsqueeze(spec_mask, -1)
      attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()  # pathが出来れば誤差伝播カット

    ph_dur = attn.sum(2)
    # duration predict
    logw_ = torch.log(ph_dur + 1e-6) * H_ph_mask
    logw = self.dp(H_ph, H_ph_mask, g=g)  # Note Normalization
    l2_dur_loss = torch.sum((logw - logw_)**2) / torch.sum(H_ph_mask)      # phoneme dur loss
    dp_H_ph = H_ph # for dur discriminator H_ph=[B, hidden, ph_len]
    dp_H_ph_mask = H_ph_mask

    # path between ph and spec
    spec_mask   = torch.unsqueeze(commons.sequence_mask(spec_lengths, spec.size(2)), 1).to(spec.dtype) # [B, 1, spec_len]
    attn_mask   = torch.unsqueeze(H_ph_mask, 2) * torch.unsqueeze(spec_mask, -1)    # attn_mask = [B, 1, ph_len, note(word)_len]
    attn_gt     = generate_path(duration=torch.unsqueeze(ph_dur,dim=1), mask=attn_mask)
    attn_gt     = torch.squeeze(attn_gt, dim=1).permute(0,2,1).float()                             # attn=[Batch, note_len,] 

    # expand prior(from ph_len to spec_len) [B, hidden, spec_len]
    H_ph_mask    = torch.matmul(H_ph_mask, attn_gt  )
    H_ph         = torch.matmul(H_ph, attn_gt       ) * spec_mask 
    H_ph_m_p     = torch.matmul(H_ph_m_p , attn_gt  ) * spec_mask 
    H_ph_logs_p  = torch.matmul(H_ph_logs_p, attn_gt) * spec_mask 

    # slice process 
    z_slice, sinewave_slice, ids_slice = commons.rand_slice_segments_with_sinewave(x=z_spec, 
                                                                                pitch=torch.squeeze(sinewave, dim=1),  # ここをSineWaveへ
                                                                                x_lengths=z_spec.size(2), 
                                                                                hop_size= self.hop_length, 
                                                                                segment_size=self.segment_size)  # frame level
    dfs_slice = commons.dfs_slice_segment(dfs=dfs,
                                          ids_str=copy.deepcopy(ids_slice),
                                          upscales=self.upsample_scales,
                                          segment_size=self.segment_size)  # frame level

    # SiFi Decoder
    voice, excitation = self.dec(x=sinewave_slice.cuda(), 
                                 c=z_slice, 
                                 d=tuple([d.to("cuda:0") for d in dfs_slice]), 
                                 g=g)

    return voice, excitation, l2_dur_loss, attn_gt, ids_slice, dp_H_ph_mask, spec_mask, \
          (z_spec, z_spec_text, H_ph_m_p, H_ph_logs_p, z_spec_m_q, z_spec_logs_q), \
          (dp_H_ph, logw, logw_)

  # batch 1 only
  def eval_infer(self, ph_IDs, ph_IDs_lengths,
                  speakerID,
                  dfs, 
                  sinewave,
                  f0_lengths,
                  noise_scale=1,
                  length_scale=1, 
                  noise_scale_w=1.):
    
    if self.n_speakers > 0:
      g = self.emb_g(speakerID).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    # text 
    H_ph,   H_ph_m_p,   H_ph_logs_p,  H_ph_mask  = self.enc_p(ph_IDs, ph_IDs_lengths, 
                                                                w_dur_ms=None,
                                                                ph_w_idx=None,
                                                                g=g)       # H_ph=[B, hidden, ph_len]

    logw = self.dp(H_ph, H_ph_mask, g=g)  # Note Normalization
    w = torch.exp(logw) * H_ph_mask * length_scale
    w = (w/torch.sum(w)) * f0_lengths  # length regurator
    w_ceil = torch.ceil(w)
    w_sum = torch.sum(w_ceil,dim=2).view(-1)
    if w_sum != f0_lengths:
       w_ceil[:,:,-1] -= w_sum - f0_lengths
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(H_ph_mask.dtype)
    attn_mask = torch.unsqueeze(H_ph_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask).squeeze(1).permute(0,2,1)

    # expand prior(from ph_len to spec_len) [B, hidden, spec_len]
    H_ph_mask    = torch.matmul(H_ph_mask  , attn)
    H_ph_m_p     = torch.matmul(H_ph_m_p   , attn) * H_ph_mask 
    H_ph_logs_p  = torch.matmul(H_ph_logs_p, attn) * H_ph_mask 

    z_spec_text = H_ph_m_p + torch.randn_like(H_ph_m_p) * torch.exp(H_ph_logs_p) * noise_scale
    z_spec = self.flow(z_spec_text, y_mask, g=g, reverse=True)

    # SiFi Decoder
    voice, _ = self.dec(x=sinewave.cuda(), 
                                 c=z_spec, 
                                 d=tuple([d.to("cuda:0") for d in dfs]), 
                                 g=g)

    return voice, attn, y_mask
  
  def get_mas_output(self,
                spec, spec_lengths,
                ph_IDs, ph_IDs_lengths,
                speakerID):
    
    if self.n_speakers > 0:
      g = self.emb_g(speakerID).unsqueeze(-1) # [B, hidden, 1]
    else:
      g = None

    # posterior encoder
    z_spec, _, _, spec_mask    = self.enc_q(spec, spec_lengths.float(), g=g)           # z_spec=[B, hidden, spec_len]

    # Flow 
    z_spec_text = self.flow(z_spec, spec_mask, g=g) # z_spec_text=[B, hidden, spec_len]

    # prior encoder 
    _,   H_ph_m_p,   H_ph_logs_p,  H_ph_mask  = self.enc_p(ph_IDs, ph_IDs_lengths, 
                                                              w_dur_ms=None,
                                                              ph_w_idx=None,
                                                              g=g)       # H_ph=[B, hidden, ph_len]

    with torch.no_grad():
      # negative cross-entropy
      s_p_sq_r = torch.exp(-2 * H_ph_logs_p) # [b, d, t]
      neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - H_ph_logs_p, [1], keepdim=True) # [b, 1, t_s]
      neg_cent2 = torch.matmul(-0.5 * (z_spec_text ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent3 = torch.matmul(z_spec_text.transpose(1, 2), (H_ph_m_p * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent4 = torch.sum(-0.5 * (H_ph_m_p ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
      neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

      epsilon = torch.std(neg_cent) * torch.randn_like(neg_cent) * self.current_mas_noise_scale
      neg_cent = neg_cent + epsilon

      attn_mask = torch.unsqueeze(H_ph_mask, 2) * torch.unsqueeze(spec_mask, -1)
      attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()  # pathが出来れば誤差伝播カット
    
    ph_dur = attn.sum(2)

    return ph_dur, attn

  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    z_p = self.flow(z, y_mask, g=g_src)
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    o_hat = self.dec(z_hat * y_mask, g=g_tgt)
    return o_hat, y_mask, (z, z_p, z_hat)

   # inference only and batch size=1 only 
  def adjust_duration(self, ph_ids, ph_dur_pd, word_dur, n_ph_in_word, noise_scale=0.33):
    ph_ids      = ph_ids[0, :]
    ph_dur_pd   = ph_dur_pd[0][0]
    word_dur    = word_dur[0]
    n_ph_in_word=n_ph_in_word[0]
    sum_duration = torch.sum(word_dur)
    total_diff = 0

    out_ph_dur  = torch.zeros_like(ph_dur_pd) # for duration output
    ph_idx = len(ph_dur_pd) 
    for idx in reversed(range(len(word_dur))):
        n_ph = n_ph_in_word[idx]
        # first
        if idx == len(word_dur)-1: 
            out_ph_dur[-1] = word_dur[idx]
            ph_idx -= 1
        # other
        else:
            target_ph_dur = ph_dur_pd[int(ph_idx-(1+z_n_ph)):int(ph_idx)] # Vowels this time + previous consonants
            adjusted_pd_dur = torch.ceil( (target_ph_dur/torch.sum(target_ph_dur) ) * word_dur[idx]) 
            if torch.sum(adjusted_pd_dur) != word_dur[idx]:
                undur = torch.sum(adjusted_pd_dur) - word_dur[idx]
                adjusted_pd_dur[0] = adjusted_pd_dur[0] - undur
            out_ph_dur[int(ph_idx-(1+z_n_ph)):int(ph_idx)] = adjusted_pd_dur
            ph_idx -= 1+z_n_ph
        z_n_ph = n_ph - 1 
    # last consonants
    if z_n_ph != 0: 
      out_ph_dur[0:z_n_ph] = ph_dur_pd[0:z_n_ph] # Vowels this time + previous consonants
    #assert torch.sum(out_ph_dur) == sum_duration  # check duration

    # adjust ph duration by statistics
    if self.ph_statistics is not False:
      diff = 0
      for idx in reversed(range(len(ph_ids))):
        ph = self.id_to_ph[int(ph_ids[idx]-1)] # minus 1 is for mask
        # fix ph_dur and calc diff
        if ph in self.ph_statistics:
          dur = out_ph_dur[idx]
          mean_ms = torch.tensor(self.ph_statistics[ph][0]).float().cuda()
          std_ms  = torch.tensor(self.ph_statistics[ph][1]).float().cuda()
          corr_ms = mean_ms + std_ms*torch.randn_like(mean_ms)*noise_scale
          corr_dur = torch.ceil(self.ms_per_frame*corr_ms)
          diff += dur - corr_dur
          out_ph_dur[idx] = corr_dur
        # shifting diff
        else:  
          out_ph_dur[idx] += diff
          diff=0

      # shifting diff last (Perhaps not necessary)
      #if diff != 0:
      #   out_ph_dur[0] += diff

      if z_n_ph != 0: 
          out_ph_dur[0:z_n_ph] = ph_dur_pd[0:z_n_ph] # Vowels this time + previous consonants
          
          #out_ph_dur[-1] -= diff # 長さ補正はconcatでやる
      #assert torch.sum(out_ph_dur) == sum_duration  # check duration
    
    # 最後に子音があれば、それだけずらす。
    if int(z_n_ph) == 0:
      total_diff = 0
    else:
      total_diff = int(torch.sum(out_ph_dur[0:int(z_n_ph)]))

    return out_ph_dur.view(1,1,-1), total_diff
    
  def encode_and_dp(self, 
                    ph_IDs, ph_IDs_lengths,
                    speakerID,
                    word_frame_dur, word_frame_dur_lenngths,
                    word_dur_ms, ph_word_flag,
                    n_ph_pool,  
                    noise_scale=1,
                    length_scale=1,
                    get_mas_output=False):

    if self.n_speakers > 0:
      g = self.emb_g(speakerID).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    # path between ph and word
    word_mask   = torch.unsqueeze(commons.sequence_mask(word_frame_dur_lenngths, word_frame_dur.size(1)), 1).to(word_frame_dur.dtype) # [B, 1, spec_len]
    ph_IDs_mask = torch.unsqueeze(commons.sequence_mask(ph_IDs_lengths, ph_IDs.size(1)), 1).to(ph_IDs.dtype) # [B, 1, ph_len]
    attn_mask   = torch.unsqueeze(word_mask, 2) * torch.unsqueeze(ph_IDs_mask, -1)    # attn_mask = [B, 1, ph_len, note(word)_len]
    attn_ph_word= generate_path(duration=torch.unsqueeze(n_ph_pool,dim=1), mask=attn_mask)
    attn_ph_word= torch.squeeze(attn_ph_word, dim=1).float()                          # attn=[Batch, note_len,] 

    word_dur_ms= torch.matmul(attn_ph_word, word_dur_ms.unsqueeze(2).float()).squeeze(2)

    # text 
    H_ph,   H_ph_m_p,   H_ph_logs_p,  H_ph_mask  = self.enc_p(ph_IDs, ph_IDs_lengths, 
                                                                w_dur_ms=word_dur_ms,
                                                                ph_w_idx=ph_word_flag,
                                                                g=g)       # H_ph=[B, hidden, ph_len]
    logw = self.dp(H_ph, H_ph_mask, g=g)  
    w = torch.exp(logw) * H_ph_mask * length_scale
    w_ceil = torch.ceil(w)

    w_ceil, diff = self.adjust_duration(ph_IDs, w_ceil, word_frame_dur, n_ph_pool)

    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(H_ph_mask.dtype)
    attn_mask = torch.unsqueeze(H_ph_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask).squeeze(1).permute(0,2,1)

    # expand prior(from ph_len to spec_len) [B, hidden, spec_len]
    H_ph_mask    = torch.matmul(H_ph_mask  , attn)
    H_ph_m_p     = torch.matmul(H_ph_m_p   , attn) * H_ph_mask 
    H_ph_logs_p  = torch.matmul(H_ph_logs_p, attn) * H_ph_mask 

    z_spec_text = H_ph_m_p + torch.randn_like(H_ph_m_p) * torch.exp(H_ph_logs_p) * noise_scale
    z_spec = self.flow(z_spec_text, y_mask, g=g, reverse=True)

    return z_spec, attn, w_ceil, g, int(diff)
  
  def synthesize(self,sinewave, z_spec, dfs, g):
    # SiFi Decoder
    voice, _ = self.dec(x=sinewave.cuda(), 
                                 c=z_spec, 
                                 d=tuple([d.to("cuda:0") for d in dfs]), 
                                 g=g)
    return voice

class VITS2_based_SiFiSinger(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    hps,
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    n_speakers,
    gin_channels,
    use_sdp=False,
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = hps["common"]["n_speaker"]
    self.gin_channels = hps["common"]["gin_channels"]
    self.transformer_flow_type = "fft" # "fft" / "mono_layer" / "pre_conv" ### When mono_layer and pre_conv, kl div loss went negative.
    self.current_mas_noise_scale = float(hps["VITS2_config"]["mas_noise_scale"])
    self.enc_gin_channels = gin_channels
  
    # VITS2 Text Encoder (Speaker Embedding)
    self.enc_p = TextEncoder_VITS2(n_vocab,hps["note_encoder"]["n_note"]+1,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=self.enc_gin_channels)

    # SiFi Decoder
    self.dec = SiFiGANGenerator(**hps["SiFiGANGenerator"])
    self.upsample_scales= hps["SiFiGANGenerator"]["upsample_scales"]

    # VITS1 Encoder
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)

    # VITS2 Flow
    self.flow = ResidualCouplingTransformersBlock(
       inter_channels, 
       hidden_channels, 
       5, 
       1, 
       4, 
       gin_channels=gin_channels, 
       use_transformer_flows=True, # Trueにすると、kl_divergense がマイナスになる。
       transformer_flow_type=self.transformer_flow_type
       )
    
    # FastSpeech2 DurationPredictor
    self.dp = VariancePredictor(hps=hps, 
                                input_size      =hps["dur_predictor"]["input_size"],
                                filter_size     =hps["dur_predictor"]["filter_size"],
                                kernel          =hps["dur_predictor"]["kernel_size"],
                                conv_output_size=hps["dur_predictor"]["filter_size"],
                                dropout         =hps["dur_predictor"]["dropout"],
                                n_speaker=hps["common"]["n_speaker"])

    if n_speakers >= 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)

    self.ms_per_frame = hps["sampling_rate"] / (hps["hop_length"] * 1000)
    self.hop_length = hps["hop_length"]
    self.oto2lab, self.ph_to_id,   self.id_to_ph, _,_ = get_g2p_dict_from_tabledata(table_path=hps["oto2lab_path"], 
                                                                                    include_converter=True)
    try:
      self.ph_statistics = torch.load(hps["ph_statistics_path"]) #{ph:[mean,var]} 
      print(f"[INFO] Loaded :", hps["ph_statistics_path"])
    except:
      self.ph_statistics = False

  def forward(self,
                spec, spec_lengths,
                ph_IDs, ph_IDs_lengths,
                ph_dur, 
                word_frame_dur, word_frame_dur_lenngths,
                word_dur_ms, ph_word_flag,
                n_ph_pool,  
                dfs, 
                sinewave,
                speakerID):
    
    if self.n_speakers > 0:
      g = self.emb_g(speakerID).unsqueeze(-1) # [B, hidden, 1]
    else:
      g = None
      
    # generate path between ph and word
    word_mask   = torch.unsqueeze(commons.sequence_mask(word_frame_dur_lenngths, word_frame_dur.size(1)), 1).to(word_frame_dur.dtype) # [B, 1, spec_len]
    ph_IDs_mask = torch.unsqueeze(commons.sequence_mask(ph_IDs_lengths, ph_IDs.size(1)), 1).to(ph_IDs.dtype) # [B, 1, ph_len]
    attn_mask   = torch.unsqueeze(word_mask, 2) * torch.unsqueeze(ph_IDs_mask, -1)    # attn_mask = [B, 1, ph_len, note(word)_len]
    attn_ph_word= generate_path(duration=torch.unsqueeze(n_ph_pool,dim=1), mask=attn_mask)
    attn_ph_word= torch.squeeze(attn_ph_word, dim=1).float()                             # attn=[Batch, note_len,] 

    # expand 
    word_dur_ms= torch.matmul(attn_ph_word, word_dur_ms.unsqueeze(2).float()).squeeze(2)
    
    # posterior encoder
    z_spec, z_spec_m_q, z_spec_logs_q, spec_mask    = self.enc_q(spec, spec_lengths.float(), g=g)           # z_spec=[B, hidden, spec_len]

    # Flow 
    z_spec_text = self.flow(z_spec, spec_mask, g=g) # z_spec_text=[B, hidden, spec_len]

    # prior encoder 
    H_ph,   H_ph_m_p,   H_ph_logs_p,  H_ph_mask  = self.enc_p(ph_IDs, ph_IDs_lengths, 
                                                              w_dur_ms=word_dur_ms,
                                                              ph_w_idx=ph_word_flag,
                                                              g=g)       # H_ph=[B, hidden, ph_len]
    
    # duration predict
    logw_ = torch.log(torch.unsqueeze(ph_dur,dim=1) + 1e-6) * ph_IDs_mask
    logw = self.dp(H_ph, ph_IDs_mask, g=g)  # Note Normalization
    l2_ph_dur_loss = torch.sum((logw - logw_)**2) / torch.sum(ph_IDs_mask)      # phoneme dur loss
    l2_word_dur_loss = torch.sum((torch.matmul(logw, attn_ph_word) - torch.matmul(logw_, attn_ph_word))**2) / torch.sum(ph_IDs_mask)     # word dur loss
    l2_full_dur_loss = (torch.sum(logw_) - torch.sum(logw) )**2 / torch.sum(ph_IDs_mask) # lengths dur loss
    l2_dur_loss = [l2_ph_dur_loss, l2_word_dur_loss, l2_full_dur_loss]

    dp_H_ph = H_ph # for dur discriminator H_ph=[B, hidden, ph_len]

    # path between ph and spec
    spec_mask   = torch.unsqueeze(commons.sequence_mask(spec_lengths, spec.size(2)), 1).to(spec.dtype) # [B, 1, spec_len]
    ph_IDs_mask = torch.unsqueeze(commons.sequence_mask(ph_IDs_lengths, ph_IDs.size(1)), 1).to(ph_IDs.dtype) # [B, 1, ph_len]
    attn_mask   = torch.unsqueeze(ph_IDs_mask, 2) * torch.unsqueeze(spec_mask, -1)    # attn_mask = [B, 1, ph_len, note(word)_len]
    attn_gt     = generate_path(duration=torch.unsqueeze(ph_dur,dim=1), mask=attn_mask)
    attn_gt     = torch.squeeze(attn_gt, dim=1).permute(0,2,1).float()                             # attn=[Batch, note_len,] 

    # expand prior(from ph_len to spec_len) [B, hidden, spec_len]
    H_ph_mask    = torch.matmul(H_ph_mask, attn_gt  )
    H_ph         = torch.matmul(H_ph, attn_gt       ) * spec_mask 
    H_ph_m_p     = torch.matmul(H_ph_m_p , attn_gt  ) * spec_mask 
    H_ph_logs_p  = torch.matmul(H_ph_logs_p, attn_gt) * spec_mask 

    # slice process 
    z_slice, sinewave_slice, ids_slice = commons.rand_slice_segments_with_sinewave(x=z_spec, 
                                                                                pitch=torch.squeeze(sinewave, dim=1),  # ここをSineWaveへ
                                                                                x_lengths=z_spec.size(2), 
                                                                                hop_size= self.hop_length, 
                                                                                segment_size=self.segment_size)  # frame level
    dfs_slice = commons.dfs_slice_segment(dfs=dfs,
                                          ids_str=copy.deepcopy(ids_slice),
                                          upscales=self.upsample_scales,
                                          segment_size=self.segment_size)  # frame level
    
    # SiFi Decoder
    voice, excitation = self.dec(x=sinewave_slice.cuda(), 
                                 c=z_slice, 
                                 d=tuple([d.to("cuda:0") for d in dfs_slice]), 
                                 g=g)
    
    return voice, excitation, l2_dur_loss, attn_gt, ids_slice, ph_IDs_mask, spec_mask, \
          (z_spec, z_spec_text, H_ph_m_p, H_ph_logs_p, z_spec_m_q, z_spec_logs_q), \
          (dp_H_ph, logw, logw_)

  # batch 1 only
  def eval_infer(self, ph_IDs, ph_IDs_lengths,
                  speakerID,
                  word_frame_dur, word_frame_dur_lenngths,
                  word_dur_ms, ph_word_flag,
                  n_ph_pool,  
                  dfs, 
                  sinewave,
                  noise_scale=1,
                  length_scale=1, 
                  noise_scale_w=1.):
    
    if self.n_speakers > 0:
      g = self.emb_g(speakerID).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    # path between ph and word
    word_mask   = torch.unsqueeze(commons.sequence_mask(word_frame_dur_lenngths, word_frame_dur.size(1)), 1).to(word_frame_dur.dtype) # [B, 1, spec_len]
    ph_IDs_mask = torch.unsqueeze(commons.sequence_mask(ph_IDs_lengths, ph_IDs.size(1)), 1).to(ph_IDs.dtype) # [B, 1, ph_len]
    attn_mask   = torch.unsqueeze(word_mask, 2) * torch.unsqueeze(ph_IDs_mask, -1)    # attn_mask = [B, 1, ph_len, note(word)_len]
    attn_ph_word= generate_path(duration=torch.unsqueeze(n_ph_pool,dim=1), mask=attn_mask)
    attn_ph_word= torch.squeeze(attn_ph_word, dim=1).float()                             # attn=[Batch, note_len,] 

    word_dur_ms= torch.matmul(attn_ph_word, word_dur_ms.unsqueeze(2).float()).squeeze(2)

    # text 
    H_ph,   H_ph_m_p,   H_ph_logs_p,  H_ph_mask  = self.enc_p(ph_IDs, ph_IDs_lengths, 
                                                                w_dur_ms=word_dur_ms,
                                                                ph_w_idx=ph_word_flag,
                                                                g=g)       # H_ph=[B, hidden, ph_len]

    logw = self.dp(H_ph, H_ph_mask, g=g)  # Note Normalization
    w = torch.exp(logw) * H_ph_mask * length_scale
    w_ceil = torch.ceil(w)

    w_ceil, diff = self.adjust_duration(ph_IDs, w_ceil, word_frame_dur, n_ph_pool)
    w_ceil[0,0,-1] -= diff # eval時だとph分余分なframeが作られるので、末尾でつじつま合わせ。

    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(H_ph_mask.dtype)
    attn_mask = torch.unsqueeze(H_ph_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask).squeeze(1).permute(0,2,1)

    # expand prior(from ph_len to spec_len) [B, hidden, spec_len]
    H_ph_mask    = torch.matmul(H_ph_mask  , attn)
    H_ph_m_p     = torch.matmul(H_ph_m_p   , attn) * H_ph_mask 
    H_ph_logs_p  = torch.matmul(H_ph_logs_p, attn) * H_ph_mask 

    z_spec_text = H_ph_m_p + torch.randn_like(H_ph_m_p) * torch.exp(H_ph_logs_p) * noise_scale
    z_spec = self.flow(z_spec_text, y_mask, g=g, reverse=True)
      
    # SiFi Decoder
    voice, _ = self.dec(x=sinewave.cuda(), 
                                 c=z_spec, 
                                 d=tuple([d.to("cuda:0") for d in dfs]), 
                                 g=g)

    return voice, attn, y_mask
    
  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    z_p = self.flow(z, y_mask, g=g_src)
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    o_hat = self.dec(z_hat * y_mask, g=g_tgt)
    return o_hat, y_mask, (z, z_p, z_hat)

  # inference only and batch size=1 only 
  def adjust_duration(self, ph_ids, ph_dur_pd, word_dur, n_ph_in_word, noise_scale=0.33):
    ph_ids      = ph_ids[0, :]
    ph_dur_pd   = ph_dur_pd[0][0]
    word_dur    = word_dur[0]
    n_ph_in_word=n_ph_in_word[0]
    sum_duration = torch.sum(word_dur)
    total_diff = 0

    out_ph_dur  = torch.zeros_like(ph_dur_pd) # for duration output
    ph_idx = len(ph_dur_pd) 
    for idx in reversed(range(len(word_dur))):
        n_ph = n_ph_in_word[idx]
        # first
        if idx == len(word_dur)-1: 
            out_ph_dur[-1] = word_dur[idx]
            ph_idx -= 1
        # other
        else:
            target_ph_dur = ph_dur_pd[int(ph_idx-(1+z_n_ph)):int(ph_idx)] # Vowels this time + previous consonants
            adjusted_pd_dur = torch.ceil( (target_ph_dur/torch.sum(target_ph_dur) ) * word_dur[idx]) 
            if torch.sum(adjusted_pd_dur) != word_dur[idx]:
                undur = torch.sum(adjusted_pd_dur) - word_dur[idx]
                adjusted_pd_dur[0] = adjusted_pd_dur[0] - undur
            out_ph_dur[int(ph_idx-(1+z_n_ph)):int(ph_idx)] = adjusted_pd_dur
            ph_idx -= 1+z_n_ph
        z_n_ph = n_ph - 1 
    # last consonants
    if z_n_ph != 0: 
      out_ph_dur[0:z_n_ph] = ph_dur_pd[0:z_n_ph] # Vowels this time + previous consonants
    #assert torch.sum(out_ph_dur) == sum_duration  # check duration

    # adjust ph duration by statistics
    if self.ph_statistics is not False:
      diff = 0
      for idx in reversed(range(len(ph_ids))):
        ph = self.id_to_ph[int(ph_ids[idx]-1)] # minus 1 is for mask
        # fix ph_dur and calc diff
        if ph in self.ph_statistics:
          dur = out_ph_dur[idx]
          mean_ms = torch.tensor(self.ph_statistics[ph][0]).float().cuda()
          std_ms  = torch.tensor(self.ph_statistics[ph][1]).float().cuda()
          corr_ms = mean_ms + std_ms*torch.randn_like(mean_ms)*noise_scale
          corr_dur = torch.ceil(self.ms_per_frame*corr_ms)
          diff += dur - corr_dur
          out_ph_dur[idx] = corr_dur
        # shifting diff
        else:  
          out_ph_dur[idx] += diff
          diff=0

      # shifting diff last (Perhaps not necessary)
      #if diff != 0:
      #   out_ph_dur[0] += diff

      if z_n_ph != 0: 
          out_ph_dur[0:z_n_ph] = ph_dur_pd[0:z_n_ph] # Vowels this time + previous consonants
          
          #out_ph_dur[-1] -= diff # 長さ補正はconcatでやる
      #assert torch.sum(out_ph_dur) == sum_duration  # check duration
    
    # 最後に子音があれば、それだけずらす。
    if int(z_n_ph) == 0:
      total_diff = 0
    else:
      total_diff = int(torch.sum(out_ph_dur[0:int(z_n_ph)]))

    return out_ph_dur.view(1,1,-1), total_diff
    
  def encode_and_dp(self, 
                    ph_IDs, ph_IDs_lengths,
                    speakerID,
                    word_frame_dur, word_frame_dur_lenngths,
                    word_dur_ms, ph_word_flag,
                    n_ph_pool,  
                    noise_scale=1,
                    length_scale=1):

    if self.n_speakers > 0:
      g = self.emb_g(speakerID).unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    # path between ph and word
    word_mask   = torch.unsqueeze(commons.sequence_mask(word_frame_dur_lenngths, word_frame_dur.size(1)), 1).to(word_frame_dur.dtype) # [B, 1, spec_len]
    ph_IDs_mask = torch.unsqueeze(commons.sequence_mask(ph_IDs_lengths, ph_IDs.size(1)), 1).to(ph_IDs.dtype) # [B, 1, ph_len]
    attn_mask   = torch.unsqueeze(word_mask, 2) * torch.unsqueeze(ph_IDs_mask, -1)    # attn_mask = [B, 1, ph_len, note(word)_len]
    attn_ph_word= generate_path(duration=torch.unsqueeze(n_ph_pool,dim=1), mask=attn_mask)
    attn_ph_word= torch.squeeze(attn_ph_word, dim=1).float()                          # attn=[Batch, note_len,] 

    word_dur_ms= torch.matmul(attn_ph_word, word_dur_ms.unsqueeze(2).float()).squeeze(2)

    # text 
    H_ph,   H_ph_m_p,   H_ph_logs_p,  H_ph_mask  = self.enc_p(ph_IDs, ph_IDs_lengths, 
                                                                w_dur_ms=word_dur_ms,
                                                                ph_w_idx=ph_word_flag,
                                                                g=g)       # H_ph=[B, hidden, ph_len]

    logw = self.dp(H_ph, H_ph_mask, g=g)  
    w = torch.exp(logw) * H_ph_mask * length_scale
    w_ceil = torch.ceil(w)

    w_ceil, diff = self.adjust_duration(ph_IDs, w_ceil, word_frame_dur, n_ph_pool)

    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(H_ph_mask.dtype)
    attn_mask = torch.unsqueeze(H_ph_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask).squeeze(1).permute(0,2,1)

    # expand prior(from ph_len to spec_len) [B, hidden, spec_len]
    H_ph_mask    = torch.matmul(H_ph_mask  , attn)
    H_ph_m_p     = torch.matmul(H_ph_m_p   , attn) * H_ph_mask 
    H_ph_logs_p  = torch.matmul(H_ph_logs_p, attn) * H_ph_mask 

    z_spec_text = H_ph_m_p + torch.randn_like(H_ph_m_p) * torch.exp(H_ph_logs_p) * noise_scale
    z_spec = self.flow(z_spec_text, y_mask, g=g, reverse=True)

    return z_spec, attn, w_ceil, g, int(diff)
  
  def synthesize(self,sinewave, z_spec, dfs, g):
    # SiFi Decoder
    voice, _ = self.dec(x=sinewave.cuda(), 
                                 c=z_spec, 
                                 d=tuple([d.to("cuda:0") for d in dfs]), 
                                 g=g)
    return voice
  
class Rezero(torch.nn.Module):
    def __init__(self):
        super(Rezero, self).__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(size=(1,)))

    def forward(self, x):
        return self.alpha * x

class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0 
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1) 
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((nn.functional.logsigmoid(z_u) + nn.functional.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw

class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None):
    x = torch.detach(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask

class DurationDiscriminator(nn.Module): #vits2
  # TODO : not using "spk conditioning" for now according to the paper.
  # Can be a better discriminator if we use it.
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    # self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    # self.norm_2 = modules.LayerNorm(filter_channels)
    self.dur_proj = nn.Conv1d(1, filter_channels, 1)

    self.pre_out_conv_1 = nn.Conv1d(2*filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.pre_out_norm_1 = modules.LayerNorm(filter_channels)
    self.pre_out_conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.pre_out_norm_2 = modules.LayerNorm(filter_channels)

    # if gin_channels != 0:
    #   self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    self.output_layer = nn.Sequential(
        nn.Linear(filter_channels, 1), 
        nn.Sigmoid() 
    )

  def forward_probability(self, x, x_mask, dur, g=None):
    dur = self.dur_proj(dur)
    x = torch.cat([x, dur], dim=1)
    x = self.pre_out_conv_1(x * x_mask)
    # x = torch.relu(x)
    # x = self.pre_out_norm_1(x)
    # x = self.drop(x)
    x = self.pre_out_conv_2(x * x_mask)
    # x = torch.relu(x)
    # x = self.pre_out_norm_2(x)
    # x = self.drop(x)
    x = x * x_mask
    x = x.transpose(1, 2)
    output_prob = self.output_layer(x)
    return output_prob

  # x=hidden_input, dur=dur
  def forward(self, x, x_mask, dur_r, dur_hat, g=None):
    x = torch.detach(x)
    # if g is not None:
    #   g = torch.detach(g)
    #   x = x + self.cond(g)
    x = self.conv_1(x * x_mask)
    # x = torch.relu(x)
    # x = self.norm_1(x)
    # x = self.drop(x)
    x = self.conv_2(x * x_mask)
    # x = torch.relu(x)
    # x = self.norm_2(x)
    # x = self.drop(x)
    
    output_probs = []
    for dur in [dur_r, dur_hat]:
      output_prob = self.forward_probability(x, x_mask, dur, g)
      output_probs.append(output_prob)
    
    return output_probs
  
class ResidualCouplingTransformersLayer(nn.Module): #vits2
  def __init__(
      self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      p_dropout=0,
      gin_channels=0,
      mean_only=False,
  ):
      assert channels % 2 == 0, "channels should be divisible by 2"
      super().__init__()
      self.channels = channels
      self.hidden_channels = hidden_channels
      self.kernel_size = kernel_size
      self.dilation_rate = dilation_rate
      self.n_layers = n_layers
      self.half_channels = channels // 2
      self.mean_only = mean_only
      #vits2
      self.pre_transformer = attentions.Encoder(
          self.half_channels,
          self.half_channels,
          n_heads=2,
          n_layers=2,
          kernel_size=3,
          p_dropout=0.1,
          window_size=None
          )

      self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
      self.enc = modules.WN_original(
          hidden_channels,
          kernel_size,
          dilation_rate,
          n_layers,
          p_dropout=p_dropout,
          gin_channels=gin_channels,
      )
      #vits2
      self.post_transformer = attentions.Encoder( 
          self.hidden_channels,
          self.hidden_channels,
          n_heads=2,
          n_layers=2,
          kernel_size=3,
          p_dropout=0.1,
          window_size=None
          )
      
      self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
      self.post.weight.data.zero_()
      self.post.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
      x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
      x0_ = self.pre_transformer(x0 * x_mask, x_mask) #vits2
      x0_ = x0_ + x0 #vits2 residual connection
      h = self.pre(x0_) * x_mask #changed from x0 to x0_ to retain x0 for the flow
      h = self.enc(h, x_mask, g=g)

      #vits2 - (experimental;uncomment the following 2 line to use)
      # h_ = self.post_transformer(h, x_mask) 
      # h = h + h_ #vits2 residual connection 

      stats = self.post(h) * x_mask
      if not self.mean_only:
          m, logs = torch.split(stats, [self.half_channels] * 2, 1)
      else:
          m = stats
          logs = torch.zeros_like(m)
      if not reverse:
          x1 = m + x1 * torch.exp(logs) * x_mask
          x = torch.cat([x0, x1], 1)
          logdet = torch.sum(logs, [1, 2])
          return x, logdet
      else:
          x1 = (x1 - m) * torch.exp(-logs) * x_mask
          x = torch.cat([x0, x1], 1)
          return x

class FFTransformerCouplingLayer(nn.Module): #vits2
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      n_layers,
      n_heads,
      p_dropout=0,
      filter_channels=768,
      mean_only=False,
      gin_channels = 0
      ):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.n_layers = n_layers
    self.half_channels = channels // 2
    self.mean_only = mean_only

    self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
    self.enc = attentions.FFT(
       hidden_channels, 
       filter_channels, 
       n_heads, 
       n_layers, 
       kernel_size, 
       p_dropout, 
       isflow = True, 
       gin_channels = gin_channels
       )
    self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels]*2, 1)
    h = self.pre(x0) * x_mask
    h_ = self.enc(h, x_mask, g=g)
    h = h_ + h
    stats = self.post(h) * x_mask
    if not self.mean_only:
      m, logs = torch.split(stats, [self.half_channels]*2, 1)
    else:
      m = stats
      logs = torch.zeros_like(m)

    if not reverse:
      x1 = m + x1 * torch.exp(logs) * x_mask
      x = torch.cat([x0, x1], 1)
      logdet = torch.sum(logs, [1,2])
      return x, logdet
    else:
      x1 = (x1 - m) * torch.exp(-logs) * x_mask
      x = torch.cat([x0, x1], 1)
      return x

class MonoTransformerFlowLayer(nn.Module): #vits2
  def __init__(
      self,
      channels,
      hidden_channels,
      mean_only=False,
  ):
    assert channels % 2 == 0, "channels should be divisible by 2"
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.half_channels = channels // 2
    self.mean_only = mean_only
    #vits2
    self.pre_transformer = attentions.Encoder(
        self.half_channels,
        self.half_channels,
        n_heads=2,
        n_layers=2,
        kernel_size=3,
        p_dropout=0.1,
        window_size=None
        )
    
    self.post = nn.Conv1d(self.half_channels, self.half_channels * (2 - mean_only), 1)
    self.post.weight.data.zero_()
    self.post.bias.data.zero_()

  def forward(self, x, x_mask, g=None, reverse=False):
    x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
    x0_ = self.pre_transformer(x0 * x_mask, x_mask) #vits2
    h = x0_ + x0 #vits2
    stats = self.post(h) * x_mask
    if not self.mean_only:
        m, logs = torch.split(stats, [self.half_channels] * 2, 1)
    else:
        m = stats
        logs = torch.zeros_like(m)
    if not reverse:
        x1 = m + x1 * torch.exp(logs) * x_mask
        x = torch.cat([x0, x1], 1)
        logdet = torch.sum(logs, [1, 2])
        return x, logdet
    else:
        x1 = (x1 - m) * torch.exp(-logs) * x_mask
        x = torch.cat([x0, x1], 1)
        return x
        
class ResidualCouplingTransformersBlock(nn.Module): #vits2
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0,
      use_transformer_flows=False,
      transformer_flow_type="pre_conv",
      ):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    if use_transformer_flows:
      if transformer_flow_type == "pre_conv":
        for i in range(n_flows):
          self.flows.append(
             ResidualCouplingTransformersLayer(
             channels, 
             hidden_channels, 
             kernel_size, 
             dilation_rate, 
             n_layers, 
             gin_channels=gin_channels, 
             mean_only=False
             )
            )
          self.flows.append(modules.Flip())
      elif transformer_flow_type == "fft":
         for i in range(n_flows):
          self.flows.append(
             FFTransformerCouplingLayer(
             channels, 
             hidden_channels, 
             kernel_size, 
             dilation_rate, 
             n_layers, 
             gin_channels=gin_channels, 
             mean_only=True
             )
            )
          self.flows.append(modules.Flip())
      elif transformer_flow_type == "mono_layer":
        for i in range(n_flows):
          self.flows.append(
             modules.ResidualCouplingLayer(
             channels, 
             hidden_channels, 
             kernel_size, 
             dilation_rate, 
             n_layers, 
             gin_channels=gin_channels, 
             mean_only=True
             )
            )
          self.flows.append(
            MonoTransformerFlowLayer(
            channels, hidden_channels, mean_only=True
            )
          )
          self.flows.append(modules.Flip())
          self.flows.append(
           MonoTransformerFlowLayer(
           channels, hidden_channels, mean_only=True
           )
           )
    else:
      for i in range(n_flows):
        self.flows.append(
           modules.ResidualCouplingLayer(
           channels, 
           hidden_channels,
           kernel_size, 
           dilation_rate, 
           n_layers, 
           gin_channels=gin_channels, 
           mean_only=True
           )
          )
        self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x
  
class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(
         modules.ResidualCouplingLayer(
         channels, 
         hidden_channels, 
         kernel_size, 
         dilation_rate, 
         n_layers, 
         gin_channels=gin_channels, 
         mean_only=True
         )
         )
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x

class unofficialRMSSinger(nn.Module):
    def __init__(self,hps,
                 vocab_size = 50,
                 ):
        super(unofficialRMSSinger, self).__init__()
        self.segment_size = hps["segments_size"] // hps["hop_length"]
        self.f0_max = int(hps["f0_max"])
        self.z_slice_segment = None
        self.eval_slice_gain =  int(hps["slice_gain"])

        self.ph_enc = TextEncoder(n_vocab=vocab_size,
                                 out_channels   =hps["ph_encoder"]["out_channels"],
                                 hidden_channels=hps["ph_encoder"]["hidden_channels"],
                                 filter_channels=hps["ph_encoder"]["filter_channels"],
                                 n_heads        =hps["ph_encoder"]["n_heads"],
                                 n_layers       =hps["ph_encoder"]["n_layers"],
                                 kernel_size    =hps["ph_encoder"]["kernel_size"],
                                 p_dropout      =hps["ph_encoder"]["p_dropout"])

        self.note_enc = NoteEncoder(n_note      =hps["note_encoder"]["n_note"]+1,
                                    hidden_channels=hps["note_encoder"]["hidden_channels"],
                                    hps         =hps)
        
        self.pooling = Pooling_from_ph_to_Word()

        self.spec_encoder = PosteriorEncoder(in_channels=hps["spec_encoder"]["spec_channels"], 
                                      out_channels=hps["spec_encoder"]["out_channels"],  
                                      hidden_channels=hps["spec_encoder"]["hidden_channels"],  
                                      kernel_size=5, 
                                      dilation_rate=1, 
                                      n_layers=16,
                                      gin_channels=hps["spec_encoder"]["gin_channels"])
        
        self.ph_note_encoder = PosteriorEncoder(in_channels=hps["ph_note_encoder"]["spec_channels"], 
                                      out_channels=hps["ph_note_encoder"]["out_channels"],  
                                      hidden_channels=hps["ph_note_encoder"]["hidden_channels"],  
                                      kernel_size=5, 
                                      dilation_rate=1, 
                                      n_layers=16,
                                      gin_channels=hps["ph_note_encoder"]["gin_channels"])

        self.flow = ResidualCouplingBlock(channels=hps["flow"]["inter_channels"], 
                                          hidden_channels=hps["flow"]["hidden_channels"], 
                                          kernel_size=5, 
                                          dilation_rate=1, 
                                          n_layers=4, 
                                          gin_channels=hps["flow"]["gin_channels"])

        self.duration_predictor = VariancePredictor(hps=hps, 
                                                    input_size      =hps["dur_predictor"]["input_size"],
                                                    filter_size     =hps["dur_predictor"]["filter_size"],
                                                    kernel          =hps["dur_predictor"]["kernel_size"],
                                                    conv_output_size=hps["dur_predictor"]["filter_size"],
                                                    dropout         =hps["dur_predictor"]["dropout"],
                                                    n_speaker=hps["common"]["n_speaker"])

        self.word_attn = WordLevelPositionalAttention(hps=hps,
                                                      word_vocab=hps["wordlevel_posattn"]["word_vocab"], # 後でちゃんとした値を入れる
                                                      emb_dim=hps["wordlevel_posattn"]["emb_dim"])
        
        self.upsampling = GaussianUpsampling(delta=hps["gaussian_upsampler"]["delta"])
        
        self.diffusion = DiffusionModels(hps=hps, vocab_size=vocab_size)

        # NSF Decoder from so-vits-svc
        self.decoder = Generator(h=hps["hifi_gan"], 
                                 sampling_rate=hps["sampling_rate"],)

        self.spk_emb = nn.Embedding(num_embeddings=hps["n_speaker"], embedding_dim=int(hps["NoisePredictor"]["inner_channels"]*5 + hps["hifi_gan"]["gin_channels"]))
        self.divide_len = int(hps["NoisePredictor"]["inner_channels"])

        self.proj = nn.Conv1d(in_channels=hps["ph_encoder"]["out_channels"]*2, 
                              out_channels=hps["ph_encoder"]["out_channels"]*2,
                              kernel_size=1)

    def forward(self,
                spec, spec_lengths,
                f0, f0_lengths, 
                ph_IDs, ph_IDs_lengths,
                noteID, noteID_lengths, 
                word_dur, word_dur_lengths,
                ph_idx_in_a_word, ph_idx_in_a_word_lengths,
                frame_word_id, frame_word_id_lengths,
                n_ph_pooling, n_ph_pooling_lengths,
                speakerID):
        
        emb_g = self.spk_emb (speakerID)
        
        # spec to z
        z, m_q, logs_q, y_mask = self.spec_encoder(spec, spec_lengths, 
                                                   g=emb_g[:, self.divide_len*2:self.divide_len*3].unsqueeze(2))
        # z to z_text
        z_p = self.flow(z, y_mask, g=emb_g[:, self.divide_len*3:self.divide_len*4].unsqueeze(2))
        
        # generate f0 mask 
        f0_mask = torch.unsqueeze(commons.sequence_mask(f0_lengths, f0.size(2)), 1).to(f0.dtype)
        word_mask = torch.unsqueeze(commons.sequence_mask(word_dur_lengths, word_dur.size(1)), 1).to(f0.dtype)

        ####################### ph_dur estimat
        # ph encode
        H_ph, H_ph_mask = self.ph_enc(ph_IDs, ph_IDs_lengths)

        # average pooling from ph to word & permute  # ph→poolingじゃなくて、
        H_word = self.pooling(ph=H_ph, 
                              ph_mask=H_ph_mask, 
                              n_ph_pooling=n_ph_pooling, 
                              word_mask=word_mask)

        # note encode ここは本物のword_durで生成する
        H_note, H_note_mask = self.note_enc(noteID, noteID_lengths, word_dur) # note_embedding Only now

        # predict word duration 
        word_dur_pd = self.duration_predictor(x      = H_note + H_word + torch.unsqueeze(emb_g[:, 0:self.divide_len],dim=1), 
                                              x_mask = torch.squeeze(H_note_mask, dim=1))
        #######################

        # word level pos attention
        H_epd, H_epd_mask = self.word_attn( hid_ph=H_ph,
                                            hid_ph_mask=H_ph_mask, 
                                            n_ph_pos=ph_idx_in_a_word,   
                                            n_ph_pos_lengths=ph_idx_in_a_word_lengths, 
                                            frame_word_id=frame_word_id, 
                                            frame_word_id_lengths=frame_word_id_lengths)

        # word level gaussian upsampler 本物のword_durを使って予測
        H_note_up = self.upsampling(hs=H_note, ds=word_dur, 
                                    h_masks=torch.squeeze(f0_mask>0,dim=1), 
                                    d_masks=torch.squeeze(H_note_mask>0,dim=1)).permute(0,2,1)

        # F0 training (possible independent training)
        loss_f0 = self.diffusion(f0         =f0,  
                                 f0_len     =f0_lengths,
                                 hid_ph     =H_epd + H_note_up, 
                                 hid_ph_mask=H_epd_mask,
                                 g          =emb_g[:, self.divide_len:self.divide_len*2])

        hidden = torch.concat([H_epd, H_note_up],dim=1)
        lengths = torch.sum(H_epd_mask, dim=2).squeeze(1)
        z_hid, m_p, logs_p, hid_mask = self.ph_note_encoder(hidden, lengths , g=emb_g[:, self.divide_len*4:self.divide_len*5].unsqueeze(2))
        #m_p, logs_p = torch.split(hidden, 192, dim=1)

        # decoder
        z_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(x=z, 
                                                                                 pitch=torch.squeeze(f0, dim=1), 
                                                                                 x_lengths=f0.size(2), 
                                                                                 segment_size=self.segment_size)

        # nsf decoder
        y_gen = self.decoder(z_slice, g=emb_g[:,self.divide_len*5:], f0=pitch_slice*1100)
        
        return y_gen, ids_slice, word_dur_pd, loss_f0, (z, z_p, m_p, logs_p, m_q, logs_q, y_mask)
    
    def inference(self, ph_IDs, ph_IDs_lengths,
                        noteID, noteID_lengths, 
                        note_dur, note_dur_lengths,
                        ph_idx_in_a_word, ph_idx_in_a_word_lengths,
                        frame_word_id, frame_word_id_lengths,
                        n_ph_pooling, n_ph_pooling_lengths,
                        speakerID,
                        f0=None, f0_lengths=None, slice_idx=None):
        assert ph_IDs.size(0) == 1
        
        emb_g = self.spk_emb (speakerID)

        # generate f0 mask 
        note_dur_mask = torch.unsqueeze(commons.sequence_mask(note_dur_lengths, note_dur.size(1)), 1).to(f0.dtype)

        # ph encode
        H_ph, H_ph_mask = self.ph_enc(ph_IDs, ph_IDs_lengths)

        # average pooling from ph to word(note) & permute
        H_word = self.pooling(ph=H_ph, 
                              ph_mask=H_ph_mask, 
                              n_ph_pooling=n_ph_pooling, 
                              word_mask=note_dur_mask)

        # note encode
        H_note, H_note_mask = self.note_enc(noteID, noteID_lengths, note_dur)

        # predict word duration
        word_dur_pd = self.duration_predictor(x      = H_note + H_word + torch.unsqueeze(emb_g[:, 0:self.divide_len],dim=1), 
                                              x_mask = torch.squeeze(H_note_mask, dim=1))
        word_dur = convert_logdur_to_intdur(word_dur_pd)
        
        # generate f0 mask 
        f0_mask = torch.ones(size=(1,int(torch.sum(word_dur)))) # 全てTrue

        # word level pos attention
        H_epd, H_epd_mask = self.word_attn( hid_ph=H_ph,
                                            hid_ph_mask=H_ph_mask, 
                                            n_ph_pos=ph_idx_in_a_word,   
                                            n_ph_pos_lengths=ph_idx_in_a_word_lengths, 
                                            frame_word_id=frame_word_id, 
                                            frame_word_id_lengths=frame_word_id_lengths)
        
        # word level gaussian upsampler
        H_note_up = self.upsampling(hs=H_note, ds=word_dur, 
                                 h_masks=torch.squeeze(f0_mask>0,dim=1), 
                                 d_masks=torch.squeeze(H_note_mask>0,dim=1)).permute(0,2,1)
        
        # F0の指定が有れば使用する
        if f0 is None or f0_lengths is None:
            # NoteIDs, NoteID_len, word_dur, speakerID = condition
            condition = [H_epd+H_note_up,  0,  0,  emb_g[:, self.divide_len:self.divide_len*2]]
            f0 = self.diffusion.sampling(condition).view(1,1,-1) * self.f0_max
            f0_mask = torch.unsqueeze(commons.sequence_mask(f0_lengths, f0.size(2)), 1).to(f0.dtype)
        else:
            f0_mask = torch.unsqueeze(commons.sequence_mask(f0_lengths, f0.size(2)), 1).to(f0.dtype)

        # nsf decoder
        z = torch.concat([H_epd, H_note_up],dim=1)
        y_gen = self.decoder(z, g=emb_g[:,self.divide_len*4:], f0=torch.unsqueeze(f0, dim=1)) 
        
        return y_gen
    
    def evaluation(self, ph_IDs, ph_IDs_lengths,
                   noteID, noteID_lengths, 
                   word_dur, word_dur_lengths, # note_dur,との連携は後で
                   ph_idx_in_a_word, ph_idx_in_a_word_lengths,
                   frame_word_id, frame_word_id_lengths,
                   n_ph_pooling, n_ph_pooling_lengths,
                   speakerID,
                   f0=None, f0_lengths=None):
        
        emb_g = self.spk_emb (speakerID)
        
        # generate mask 
        #note_dur_sum = torch.sum(note_dur)
        #note_mask = commons.sequence_mask(note_dur_sum.view(1,-1), note_dur_sum).to(f0.dtype).cuda()
        #note_mask = commons.sequence_mask(word_dur_lengths, word_dur.size(1)).to(f0.dtype)

        # ph encode
        H_ph, H_ph_mask = self.ph_enc(ph_IDs, ph_IDs_lengths)

        # average pooling from ph to word & permute
        word_mask = torch.unsqueeze(commons.sequence_mask(word_dur_lengths, word_dur.size(1)), 1).to(f0.dtype)
        H_word = self.pooling(ph=H_ph, 
                              ph_mask=H_ph_mask, 
                              n_ph_pooling=n_ph_pooling, 
                              word_mask=word_mask)          # [B, word_len, hidden]

        # note encode
        H_note, H_note_mask = self.note_enc(noteID, noteID_lengths, word_dur)   # [B, word_len, hidden]

        # predict word duration
        word_dur_pd = self.duration_predictor(x      = H_note.detach() + H_word.detach() + torch.unsqueeze(emb_g[:, 0:self.divide_len],dim=1), 
                                              x_mask = torch.squeeze(H_note_mask, dim=1))
        
        # word level pos attention
        H_epd, H_epd_mask = self.word_attn( hid_ph=H_ph,
                                            hid_ph_mask=H_ph_mask, 
                                            n_ph_pos=ph_idx_in_a_word,   
                                            n_ph_pos_lengths=ph_idx_in_a_word_lengths, 
                                            frame_word_id=frame_word_id,                    # ここをword_dur_pdで埋める必要アリ
                                            frame_word_id_lengths=frame_word_id_lengths)
        
        # word level gaussian upsampler
        H_note_up = self.upsampling(hs=H_note, ds=word_dur,  ### 分岐01、ここは予測値を使うこともできる
                                 h_masks=torch.squeeze(H_epd_mask>0,dim=1), 
                                 d_masks=torch.squeeze(H_note_mask>0,dim=1)).permute(0,2,1)
        
        # F0 sampling NoteIDs, NoteID_len, NoteDur, speakerID = condition
        condition = [H_epd+H_note_up,  noteID_lengths,  word_dur,  emb_g[:, self.divide_len:self.divide_len*2]]
        #condition = [H_epd+H_note_up,  noteID_lengths,  convert_logdur_to_intdur(word_dur_pd) ,  emb_g[:, self.divide_len:self.divide_len*2]]
        f0_pd = self.diffusion.sampling(condition).view(1,1,-1) * self.f0_max
        #f0_pd_mask = torch.unsqueeze(commons.sequence_mask(f0_lengths, f0.size(2)), 1).to(f0.dtype)
        #f0_mask = torch.unsqueeze(commons.sequence_mask(f0_lengths, f0.size(2)), 1).to(f0.dtype)

        # decoder
        
        hidden = torch.concat([H_epd, H_note_up],dim=1)
        lengths = torch.sum(H_epd_mask, dim=2).squeeze(1)
        z_hid, m_p, logs_p, hid_mask = self.ph_note_encoder(hidden, lengths , g=emb_g[:, self.divide_len*4:self.divide_len*5].unsqueeze(2))
        #m_p, logs_p = torch.split(hidden, 192, dim=1)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * 1 # noisescale
        z = self.flow(z_p, H_epd_mask, g=emb_g[:, self.divide_len*3:self.divide_len*4].unsqueeze(2), reverse=True)
        
        ### 分岐02
        # for eval with f0gt
        z_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(x=z, 
                                                                                 pitch=torch.squeeze(f0, dim=1), 
                                                                                 x_lengths=f0.size(2), 
                                                                                 segment_size=self.segment_size*self.eval_slice_gain)
        y_gen_f0gt = self.decoder(z_slice, g=emb_g[:,self.divide_len*5:], f0=pitch_slice*1100) # nsf decoder

        # for eval with f0pd
        z_slice, pitch_slice, ids_slice = commons.slice_segments_with_pitch(x=z, 
                                                                            pitch=torch.squeeze(f0_pd, dim=1), 
                                                                            ids_str=ids_slice, 
                                                                            segment_size=self.segment_size*self.eval_slice_gain)
        y_gen_f0pd = self.decoder(z_slice, g=emb_g[:,self.divide_len*5:], f0=pitch_slice*1100) # nsf decoder

        return y_gen_f0gt, y_gen_f0pd, f0_pd, ids_slice
    
# Copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
class GaussianUpsampling(torch.nn.Module):
    """Gaussian upsampling with fixed temperature as in:

    https://arxiv.org/abs/2010.04301

    """

    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta # deltaは1/(2*sigma^2)の値だと思われる

    def forward(self, hs, ds, h_masks=None, d_masks=None):
        """Upsample hidden states according to durations.

        Args:
            hs (Tensor): Batched hidden state to be expanded (B, T_text, adim).
            ds (Tensor): Batched token duration (B, T_text).
            h_masks (Tensor): Mask tensor (B, T_feats).
            d_masks (Tensor): Mask tensor (B, T_text).

        Returns:
            Tensor: Expanded hidden state (B, T_feat, adim).

        """
        B = ds.size(0)
        device = ds.device

        if ds.sum() == 0:
            logging.warning(
                "predicted durations includes all 0 sequences. "
                "fill the first element with 1."
            )
            # NOTE(kan-bayashi): This case must not be happened in teacher forcing.
            #   It will be happened in inference with a bad duration predictor.
            #   So we do not need to care the padded sequence case here.
            ds[ds.sum(dim=1).eq(0)] = 1

        if h_masks is None:
            T_feats = ds.sum().int()
        else:
            T_feats = h_masks.size(-1)
        t = torch.arange(0, T_feats).unsqueeze(0).repeat(B, 1).to(device).float()   # [B, frame]
        if h_masks is not None:
            t = t * h_masks.float()

        c = ds.cumsum(dim=-1) - ds / 2 # ds.cumsumはeのこと
        energy = -1 * self.delta * (t.unsqueeze(-1) - c.unsqueeze(1)) ** 2 # energy = w_t^n
        if d_masks is not None:
            #energy = energy.masked_fill(
            #    ~(d_masks.unsqueeze(1).repeat(1, T_feats, 1)), -float("inf")
            #)
            d_masks = d_masks.unsqueeze(1).repeat(1, T_feats, 1)
            energy = energy.masked_fill(d_masks==0, -float("inf"))

        p_attn = torch.softmax(energy, dim=2)  # (B, T_feats, T_text)、これが重み
        hs = torch.matmul(p_attn, hs) # 重み適応 hs=[B, T_feats, hidden]
        return hs

class Pooling_from_ph_to_Word(nn.Module):
    def __init__(self,):
        super(Pooling_from_ph_to_Word, self).__init__()
    def forward(self,ph,            # [B, hidden,   ph_len]
                     ph_mask,       # [B,      1,   ph_len]
                     n_ph_pooling,  # [B, word_len]
                     word_mask):    # [B,      1,   word_len]
        attn_mask = torch.unsqueeze(word_mask, 2) * torch.unsqueeze(ph_mask, -1)            # attn_mask= [B, 1, ph_len, word_len]
        attn = generate_path(duration=torch.unsqueeze(n_ph_pooling,dim=1), mask=attn_mask ) # attn= [B, 1, ph_len, word_len]
        attn = torch.squeeze(attn, dim=1)
        attn = torch.div(attn, torch.unsqueeze(n_ph_pooling, dim=1)) # NaN occurs
        attn = torch.nan_to_num(attn) # replace NaN with 0.
        H_word = torch.matmul(ph, attn.detach()) # detach and pooling process 
        return H_word.permute(0,2,1)

class WordLevelPositionalAttention(nn.Module):
    def __init__(self,
                 hps,
                 word_vocab,
                 emb_dim
                 ):
        super(WordLevelPositionalAttention, self).__init__()
        self.n_ph_emb = nn.Embedding(num_embeddings=5, embedding_dim=int(emb_dim//4)) 
        self.word_id_emb = nn.Embedding(num_embeddings=word_vocab, embedding_dim=emb_dim) 

        self.linear_proj = nn.Conv1d(in_channels=emb_dim + int(emb_dim//4),
                                     out_channels=emb_dim, kernel_size=1)
        self.emb_dim = emb_dim
        
    def forward(self, 
                hid_ph,                 # [B, hidden, ph_len]
                hid_ph_mask,            # [B,      1, ph_len]
                n_ph_pos,               # [B, ph_len]
                n_ph_pos_lengths,       # [B,] 
                frame_word_id,          # [B, frame]
                frame_word_id_lengths): # [B, ]
        
        ph_len = hid_ph.size(2)
        frame_len = frame_word_id.size(1)

        n_ph_mask = torch.unsqueeze(commons.sequence_mask(n_ph_pos_lengths, n_ph_pos.size(1)), 1).to(n_ph_pos.dtype) # [B, 1, ph_len]
        emb_n_ph_pos = self.n_ph_emb(n_ph_pos).permute(0,2,1) # emb_n_ph_pos=[B, temp_hidden, ph_len]
        emb_n_ph_pos = emb_n_ph_pos * n_ph_mask

        frame_word_id_mask = torch.unsqueeze(commons.sequence_mask(frame_word_id_lengths, frame_word_id.size(1)), 1).to(frame_word_id.dtype) #[B, 1, frame]
        emb_frame_word_id = self.word_id_emb(frame_word_id).permute(0,2,1) # [B, hidden, frame]
        emb_frame_word_id = emb_frame_word_id * frame_word_id_mask
        
        H_k = torch.cat([hid_ph, emb_n_ph_pos], dim=1)
        H_k = self.linear_proj(H_k)
        H_k  = H_k.permute(0,2,1) # [B, ph_len, hidden]

        attn_score = torch.matmul( H_k, emb_frame_word_id) / math.sqrt(self.emb_dim) # attn_score = [B, ph_len, frame]
        attn_score = attn_score.masked_fill(n_ph_mask.squeeze(1).unsqueeze(2).repeat(1, 1, frame_len)==0, -float("inf"))
        attn_score = attn_score.masked_fill(frame_word_id_mask.repeat(1, ph_len, 1)==0, -float("1e20")) # NaN対策
        attn_score = torch.softmax(attn_score, dim=1) 

        H_epd = torch.matmul(hid_ph, attn_score)
        return H_epd * frame_word_id_mask, frame_word_id_mask        

class NoteEncoder(nn.Module):
    def __init__(self,
                 n_note,
                 hidden_channels,
                 hps,
                 ):
        super(NoteEncoder, self).__init__()
        self.hidden_ch = hidden_channels
        #self.note_emb = nn.Embedding(num_embeddings=n_note, 
        #                             embedding_dim=hidden_channels)
        self.dur_emb  = nn.Linear   (in_features=1,  # max_len
                                     out_features=hidden_channels)
        self.note_emb  = nn.Linear   (in_features=1,  # max_len
                                     out_features=hidden_channels)
        self.hidden_channels=hidden_channels
        
        ### wordIDをselfAttnして加算してもよいかも
    def forward(self, noteID, noteID_lengths, note_dur=None):
        noteID_mask = torch.unsqueeze(commons.sequence_mask(noteID_lengths, noteID.size(1)), 1).to(noteID.dtype)
        
        emb = self.note_emb(noteID.unsqueeze(2)) / math.sqrt(self.hidden_ch) 
        if note_dur is not None:
          emb += self.dur_emb(note_dur.unsqueeze(2)) / math.sqrt(self.hidden_ch) 
        
        return emb.permute(0,2,1), noteID_mask


####################
# from FastSpeech2 #
####################
class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, hps,
                 input_size,
                 filter_size,
                 kernel,
                 conv_output_size,
                 dropout,
                 n_speaker):
        super(VariancePredictor, self).__init__()

        self.input_size         = input_size
        self.filter_size        = filter_size
        self.kernel             = kernel
        self.conv_output_size   = conv_output_size
        self.dropout            = dropout
        

        self.emb_g = nn.Conv1d(input_size, self.filter_size, kernel_size=1)

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    #("relu_2", nn.ReLU()),
                    ("relu_2", nn.Mish()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    #("relu_2", nn.ReLU()),
                    ("relu_2", nn.Mish()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + self.emb_g(g)
        x = self.conv_layer(x.permute(0,2,1))
        x = self.linear_layer(x)
        x = x.permute(0,2,1)

        if x_mask is not None:
            x = x.masked_fill(x_mask==0, 0.0)

        return x


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

class DiffusionModels(nn.Module):
    def __init__(self,
                 hps,
                 vocab_size,
                 training=True):
        super(DiffusionModels, self).__init__()

        self.num_timesteps = hps["Diffusion"]["T"]
        if training is True:
          self.infer_timesteps = hps["Diffusion"]["T"]
          self.ddim = False
        else:
          self.infer_timesteps = hps["Diffusion"]["N"]
          self.ddim = hps["Diffusion"]["ddim"]

        self.f0_max = float(hps["f0_max"])
        self.lf0_max = 2595. * torch.log10(1. + torch.tensor(hps["f0_max"]).float() / 700.) / 500

        # F0 diffusion modules
        self.f0_diff = F0_Diffusion(hps=hps)

        # Voice/UnVoice diffusion modules
        #self.vuv_diff = MultinomialDiffusion(num_classes=3,   # vとuvとmaskの3通り
        #                                     timesteps=self.num_timesteps,
        #                                     loss_type="vb_stochastic",
        #                                     parametrization="x0")
        self.noise_schedule = None
        self.N = None 
        self.step_infer = None

        # Noise Predictor
        self.noise_predictor = NoisePredictor(hps=hps, 
                                              out_channels            = hps["NoisePredictor"]['out_channels'],
                                              inner_channels          = hps["NoisePredictor"]['inner_channels'],
                                              WN_in_channels          = hps["NoisePredictor"]['WN_in_channels'],
                                              WN_kernel_size          = hps["NoisePredictor"]['WN_kernel_size'],
                                              WN_dilation_rate        = hps["NoisePredictor"]['WN_dilation_rate'],
                                              WN_n_layers             = hps["NoisePredictor"]['WN_n_layers'],
                                              WN_p_dropout            = hps["NoisePredictor"]['WN_p_dropout'],
                                              Attn_filter_channels    = hps["NoisePredictor"]['Attn_filter_channels'],
                                              Attn_kernel_size        = hps["NoisePredictor"]['Attn_kernel_size'],
                                              Attn_n_layers           = hps["NoisePredictor"]['Attn_n_layers'],
                                              Attn_n_heads            = hps["NoisePredictor"]['Attn_n_heads'],
                                              Attn_p_dropout          = hps["NoisePredictor"]['Attn_p_dropout'],
                                              Attn_vocab_size         = vocab_size+1, # mask用に0を取っておく。
                                              n_speakers              = hps["NoisePredictor"]['n_speakers'],
                                              Diff_step_embed_in      = hps["NoisePredictor"]['Diff_step_embed_in'],
                                              Diff_step_embed_mid     = hps["NoisePredictor"]['Diff_step_embed_mid'],
                                              Diff_step_embed_out     = hps["NoisePredictor"]['Diff_step_embed_out'],
                                              gin_channels            = hps["NoisePredictor"]['n_speakers'])

    # 入力を受け取り、ロスを返す。
    def forward(self, f0,  f0_len,
                      IDs, IDs_len, IDs_dur,
                      NoteIDs = None, NoteID_len = None,
                      g=None):
        
        f0 = self.norm_f0(f0)
        if NoteIDs is not None:
          NoteIDs = self.norm_f0(NoteIDs)

        # Generate timesteps 
        B, _, _ = f0.shape  
        timesteps= torch.randint(self.num_timesteps, size=(B, 1, 1)).cuda()
        #timesteps, pt = self.vuv_diff.sample_time(B, device="cuda", method='importance')

        # F0 preprocess (F0 → NoisyF0)
        noisy_f0, noise_gt = self.f0_diff.get_noisy_f0(f0=f0, ts=timesteps)

        # vuv preprocess (VUV → NoisyVUV)
        # noisy_vuv, vuv_diff_params = self.vuv_diff.preprocess(x=vuv, t_int=timesteps.view(-1))

        # Noise prediction 
        f0_noise = self.noise_predictor(f0=noisy_f0,  
                                        f0_len=f0_len,    # noisyF0
                                        ph_IDs=IDs, 
                                        ph_IDs_len=IDs_len,   # Condition (ph)
                                        ph_ID_dur=IDs_dur, 
                                        NoteIDs=NoteIDs, 
                                        NoteIDS_len=NoteID_len,
                                        timesteps=timesteps.view(B, 1),
                                        g=g)

        # Loss calculation for F0
        loss_f0 = self.f0_diff.get_loss( f0_noise_pd=f0_noise, 
                                         noisy_f0=noisy_f0, 
                                         noise_gt=noise_gt)

        # Loss calculation for VUV
        #loss_vuv = self.vuv_diff.postprocess(model_out=vuv_noise, 
        #                                     parameters=vuv_diff_params,
        #                                     t_int=timesteps.view(-1), 
        #                                     t_float=pt)
        #loss_vuv = loss_vuv.sum() / (math.log(2) * torch.sum(vuv_len))

        return loss_f0  #, loss_vuv 
    
    @torch.inference_mode()
    def sampling(self, condition):
        #ph_IDs, ph_IDs_len, ph_IDs_dur, NoteIDs, NoteID_len, NoteDur,speakerID = condition
        f0_len = int(torch.sum(condition[2][0]))

        if self.noise_schedule is None:
          self.noise_schedule = self.f0_diff.get_noise_schedule(timesteps=self.infer_timesteps, training=False) 
        # sampling using DDPM reverse process. 
        #f0_pd, vuv_pd = self.sampling_given_noise_schedule(size=(1, 1, f0_len),  
        #                                                   inference_noise_schedule=noise_schedule,
        #                                                   condition=condition, 
        #                                                   ddim=False, 
        #                                                   return_sequence=False)
        f0_pd = self.sampling_given_noise_schedule(size=(1, 1, f0_len),  
                                                           inference_noise_schedule=self.noise_schedule,
                                                           condition=condition, 
                                                           ddim=self.ddim, 
                                                           return_sequence=False)

        return self.reverse_norm_f0(f0_pd)
        #return f0_pd, vuv_pd
    
    @torch.inference_mode()
    def sampling_given_noise_schedule(
        self,
        size,
        inference_noise_schedule,
        condition=None,
        ddim=False,
        return_sequence=False):
        """
        Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

        Parameters:
        net (torch network):            the wavenet models
        size (tuple):                   size of tensor to be generated,
                                        usually is (number of audios to generate, channels=1, length of audio)
        diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                        note, the tensors need to be cuda tensors
        condition (torch.tensor):       ground truth mel spectrogram read from disk
                                        None if used for unconditional generation

        Returns:
        the generated audio(s) in torch.tensor, shape=size
        """

        #IDs_gt, IDs_gt_len, dur_gt, speaker_ID = condition
        ph_IDs, ph_IDs_len, ph_IDs_dur, NoteIDs, NoteID_len, _, speakerID = condition
        if NoteIDs is not None:
          NoteIDs = self.norm_f0(NoteIDs)
        N , steps_infer = self.f0_diff.get_step_infer(size=size,
                                                     inference_noise_schedule=inference_noise_schedule)
          
        x = self.f0_diff.generate_x(size).cuda()
        #log_z = self.vuv_diff.generate_log_z(size).cuda()

        if return_sequence:
            x_ = copy.deepcopy(x)
            xs = [x_]

        # need  N と steps_infer
        with torch.no_grad():
            for n in tqdm(range(N - 1, -1, -1),desc="Sampling..."):
                diffusion_steps = (steps_infer[n] * torch.ones((size[0], 1))).cuda()

                # Multinomial diffusion preprocess
                #model_in, log_z = self.vuv_diff.sample_preprocess(log_z=log_z, t=n)

                # Noise Prediction
                #f0_noise, vuv_noise = self.noise_predictor(f0=x,  
                #                                           f0_len=torch.tensor([dur_gt.size(1)], dtype=torch.int64, device="cuda:0"),
                #                                           IDs=IDs_gt, 
                #                                           IDs_len=IDs_gt_len, 
                #                                           vuv=model_in,                  
                #                                           vuv_len=torch.tensor([dur_gt.size(1)], dtype=torch.int64, device="cuda:0"),               
                #                                           attn=dur_gt, 
                #                                           timesteps=diffusion_steps,
                #                                           g=speaker_ID)
                
                f0_noise = self.noise_predictor(f0=x,  
                                                f0_len=torch.tensor([size[2]], dtype=torch.int64, device="cuda:0"),    # noisyF0
                                                ph_IDs=ph_IDs, 
                                                ph_IDs_len=ph_IDs_len,   # Condition (ph)
                                                ph_ID_dur=ph_IDs_dur, 
                                                timesteps=diffusion_steps,
                                                NoteIDs=NoteIDs, 
                                                NoteIDS_len=NoteID_len,
                                                g=speakerID)

                # Multinomial diffusion postprocess
                #log_z = self.vuv_diff.sample_postprocess(model_out=vuv_noise, log_z=log_z, t_int=n)

                # Denoising
                x = self.f0_diff.denoising(x=x, noise_pd=f0_noise, ddim=ddim, n=n)

                if return_sequence:
                    x_ = copy.deepcopy(x)
                    xs.append(x_)

            # VUV Decoding
            #vuv_pd = self.vuv_diff.decode_log_z(log_z)

        if return_sequence:
            return xs
        return x
        #return x, vuv_pd
    def norm_f0(self,f0):
        #lf0 = 2595. * torch.log10(1. + f0 / 700.) / 500 / self.lf0_max
        f0 = f0 / self.f0_max
        return f0 
    def reverse_norm_f0(self,f0):
        #lf0[lf0<0] = 0
        #f0 = (700 * (torch.pow(10, lf0 *self.lf0_max * 500 / 2595) - 1))
        f0[f0<0] = 0
        f0 = f0 * self.f0_max
        return f0  

class NoisePredictor(nn.Module):
    def __init__(self,hps,
        out_channels,
        inner_channels,
        WN_in_channels,
        WN_kernel_size,
        WN_dilation_rate,
        WN_n_layers,
        WN_p_dropout,
        Attn_filter_channels,
        Attn_kernel_size,
        Attn_n_layers,
        Attn_n_heads,
        Attn_p_dropout,
        Attn_vocab_size,
        n_speakers,
        Diff_step_embed_in,
        Diff_step_embed_mid,
        Diff_step_embed_out,
        gin_channels=0):
        super().__init__()

        self.out_channels=out_channels
        self.diffusion_step_embed_dim_in = Diff_step_embed_in

        # Timestep Embedding
        self.fc_t = nn.ModuleList()
        self.fc_t1 = nn.Linear(Diff_step_embed_in, Diff_step_embed_mid)
        self.fc_t2 = nn.Linear(Diff_step_embed_mid, Diff_step_embed_out)

        # Speaker / VUV Embedding
        #self.emb_uv = nn.Embedding(3, inner_channels) # vとuvとmaskの３つ
        self.emb_g  = nn.Embedding(n_speakers, gin_channels)

        # NoteEncoder
        self.note_enc = NoteEncoder(n_note=hps["note_encoder"]["n_note"]+1, 
                                    hidden_channels=hps["note_encoder"]["hidden_channels"],
                                    hps=hps)

        # Attention
        self.text_encoder = TextEncoder(n_vocab         =Attn_vocab_size,
                                        out_channels    =inner_channels,
                                        hidden_channels =inner_channels,
                                        filter_channels =Attn_filter_channels,
                                        n_heads         =Attn_n_heads,
                                        n_layers        =Attn_n_layers,
                                        kernel_size     =Attn_kernel_size,
                                        p_dropout       =Attn_p_dropout   )

        # WaveNet
        self.pre = nn.Conv1d(WN_in_channels, inner_channels, 1)
        self.WaveNet = modules.WN(hidden_channels=inner_channels,
                                  Diff_step_embed_out=Diff_step_embed_out,
                                  kernel_size=WN_kernel_size,
                                  dilation_rate=WN_dilation_rate, 
                                  n_layers=WN_n_layers, 
                                  gin_channels=gin_channels,
                                  p_dropout=WN_p_dropout)

        #self.rezero = Rezero() 

        # projection
        output_ch = 1 # 4 = [f0 vuv]
        self.proj_1 = nn.Conv1d(inner_channels, output_ch, 1) 
        self.relu = nn.Mish()
        self.proj_2 = nn.Conv1d(output_ch, output_ch, 1) 
        #self.vuv_proj = nn.Conv1d(1,4,1)

    def forward(self,   f0,  
                        f0_len,    # noisyF0
                        ph_IDs, 
                        ph_IDs_len,   # Condition (ph)
                        ph_ID_dur, 
                        #vuv, vuv_len, 
                        timesteps,
                        NoteIDs=None, 
                        NoteIDS_len=None, 
                        g=None):

        # Embedding timesteps
        emb_timesteps = calc_diffusion_step_embedding(timesteps, self.diffusion_step_embed_dim_in)
        emb_timesteps = swish(self.fc_t1(emb_timesteps))
        emb_timesteps = swish(self.fc_t2(emb_timesteps))

        # Embedding speakerID 
        g = torch.unsqueeze(self.emb_g(g),dim=2) # [Batch, n_speakers] to [Batch, 1, gin_channels]

        # Projection 
        f0_mask = torch.unsqueeze(commons.sequence_mask(f0_len, f0.size(2)), 1).to(f0.dtype)
        f0 = self.pre(f0) * f0_mask                 # [Batch, Channel=1, f0_len] to [Batch, inner_channels, f0_len]

        # Embedding vuv labels
        #vuv_mask = torch.unsqueeze(commons.sequence_mask(vuv_len, vuv.size(1)), 1).to(vuv.dtype)
        #vuv = self.emb_uv(vuv).permute(0,2,1) * vuv_mask    # [Batch, f0_len]     to [Batch, inner_channel, f0_len]


        # Encoding (output is masked) Attention Embedding
        ph_IDs, ph_IDs_mask = self.text_encoder(ph_IDs, ph_IDs_len)                 # [Batch, inner_channel, IDs_len]   

        if NoteIDs is not None:
          noteIDs, noteIDs_mask = self.note_enc(  noteID=NoteIDs, 
                                                noteID_lengths=NoteIDS_len)

        # expand ph_len to f0_len 
        attn_mask = torch.unsqueeze(ph_IDs_mask, 2) * torch.unsqueeze(f0_mask, -1)
        attn = generate_path(duration=torch.unsqueeze(ph_ID_dur,dim=1), mask=attn_mask )
        attn = torch.squeeze(attn, dim=1).permute(0,2,1)                          # [Batch, IDs_len, f0_len] 
        ph_IDs = torch.matmul(ph_IDs, torch.tensor(attn, dtype=torch.float32))    # to [Batch, inner_channel, f0_len] 
        
        if NoteIDs is not None:
          noteIDs = torch.matmul(noteIDs, torch.tensor(attn, dtype=torch.float32))  # to [Batch, inner_channel, f0_len] 
          hidden = ph_IDs + noteIDs
        else:
          hidden = ph_IDs
        # NoisePrediction Process
        f0 = self.WaveNet(  x         =f0 ,#+ vuv,      
                            x_mask    =f0_mask, 
                            IDs       =hidden, 
                            IDs_mask  =ph_IDs_mask, 
                            timesteps =emb_timesteps, 
                            g         =g)  

        # Projection
        f0 = self.proj_1(f0)
        f0 = self.relu  (f0)
        f0 = self.proj_2(f0) * f0_mask

        # ReZero Regularization
        # f0_vuv = self.rezero(f0_vuv)
        
        # f0, vuv_vector = torch.split(tensor=f0_vuv, 
        #                             split_size_or_sections=1,  # ここlabel３つ
        #                             dim=1 )
        # vuv_vector = self.vuv_proj(vuv_vector)
        

        return f0 # , torch.zeros(size=(f0.size(0), 3, f0.size(2))).cuda()


class U_Net_1D(torch.nn.Module):
    def __init__(self,in_ch=192,             # Hidden
                    inner_ch=256,          # 
                    filter_ch=768,
                    out_ch=4,              # f0, mask, v, uv  
                    time_embed_dim=512,
                    attn_filter_ch=768,
                    ):
        super().__init__()
        
        ########################
        ### For Down Process ###
        ########################
        self.ResBlock_D1 = ResBlock1D(inp_channels=in_ch, 
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.CrossAttn_D1 = CrossAttn(hidden_channels=inner_ch,
                                      filter_channels=filter_ch,
                                      n_heads=4,
                                      n_layers=1,
                                      kernel_size=3,
                                      p_dropout=0.1)
        self.DownSample_D1 =Downsample1D(channels=inner_ch,
                                         use_conv=True,
                                         out_channels=inner_ch)

        self.ResBlock_D2 = ResBlock1D(inp_channels=inner_ch, 
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.CrossAttn_D2 = CrossAttn(hidden_channels=inner_ch,
                                      filter_channels=filter_ch,
                                      n_heads=4,
                                      n_layers=1,
                                      kernel_size=3,
                                      p_dropout=0.1)
        self.DownSample_D2 =Downsample1D(channels=inner_ch,
                                         use_conv=True,
                                         out_channels=inner_ch)
        
        
        #######################
        ### For Mid Process ###
        #######################
        self.ResBlock_M1 = ResBlock1D(inp_channels=inner_ch, 
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.SelfAttn    = SelfAttn(hidden_channels=inner_ch,
                                    filter_channels=filter_ch,
                                    n_heads=4,
                                    n_layers=1,
                                    kernel_size=3,
                                    p_dropout=0.1)
        self.ResBlock_M2 = ResBlock1D(inp_channels=inner_ch, 
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)

        
        ######################
        ### For Up Process ###
        ######################
        self.UpSample_U1 = Upsample1D(channels=inner_ch,
                                      use_conv=True,
                                      out_channels=inner_ch)
        self.ResBlock_U1 = ResBlock1D(inp_channels=inner_ch*2, 
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.CrossAttn_U1 = CrossAttn(hidden_channels=inner_ch,
                                      filter_channels=filter_ch,
                                      n_heads=4,
                                      n_layers=1,
                                      kernel_size=3,
                                      p_dropout=0.1)
        
        self.UpSample_U2 = Upsample1D(channels=inner_ch,
                                      use_conv=True,
                                      out_channels=inner_ch)
        self.ResBlock_U2 = ResBlock1D(inp_channels=in_ch*2, 
                                      out_channels=inner_ch,
                                      time_embed_dim=time_embed_dim)
        self.CrossAttn_U2 = CrossAttn(hidden_channels=inner_ch,
                                      filter_channels=filter_ch,
                                      n_heads=4,
                                      n_layers=1,
                                      kernel_size=3,
                                      p_dropout=0.1)
        
        self.final = OutConv1DBlock(num_groups_out=8,
                                    out_channels=out_ch,
                                    embed_dim=inner_ch,
                                    act_fn="mish")
        
    
    def forward(self, x, x_mask, h, h_mask, t):

        # Down Process 
        x = self.ResBlock_D1(inputs=x, t=t)
        x_0 = self.CrossAttn_D1(x=x, x_mask=x_mask, h=h, h_mask=h_mask)
        _, _, x_0_len = x_0.shape
        x = self.DownSample_D1(inputs=x_0) # length/2 
        
        x = self.ResBlock_D2(inputs=x, t=t)
        x_mask_1 = self.mask_downsample(downsample_x=x, mask=x_mask)
        x_1 = self.CrossAttn_D2(x=x, x_mask=x_mask_1, h=h, h_mask=h_mask)
        _, _, x_1_len = x_1.shape
        x = self.DownSample_D2(inputs=x_1) # length/4

        # Mid Process
        x = self.ResBlock_M1(inputs=x, t=t)
        x_mask_2 = self.mask_downsample(downsample_x=x, mask=x_mask_1)
        x = self.SelfAttn(x=x,x_mask=x_mask_2)
        x = self.ResBlock_M2(inputs=x, t=t) # length/4

        # Up Process
        x = self.UpSample_U1(inputs=x) # length/2
        x = torch.cat(tensors=[x_1, x[:,:,:x_1_len]], dim=1) # concat in "channel direction"
        x = self.ResBlock_U1(inputs=x, t=t)
        x = self.CrossAttn_U1(x=x, x_mask=x_mask_1, h=h, h_mask=h_mask)

        x = self.UpSample_U2(inputs=x)
        x = torch.cat(tensors=[x_0, x[:,:,:x_0_len]], dim=1) # concat in "channel direction"
        x = self.ResBlock_U2(inputs=x, t=t)
        x = self.CrossAttn_U2(x=x, x_mask=x_mask, h=h, h_mask=h_mask)

        # Final Process
        x = self.final(x)
        return x
    
    def mask_downsample(self, downsample_x, mask):
        _,_,mask_length = mask.shape
        _,_,x_length = downsample_x.shape
        if mask_length % 2 == 0: # even 
            mask = mask[:,:,1::2]
        else:
            mask = mask[:,:,::2]

        if x_length == mask.size(2):
            return mask

def swish(x):
    return x * torch.sigmoid(x)

class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN_original(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask

class TextEncoder_VITS2(nn.Module):
  def __init__(self,
      n_vocab,
      n_note, 
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      gin_channels=0):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels
    self.x_emb = nn.Embedding(n_vocab, hidden_channels)
    self.ph_w_idx_emb = nn.Embedding(2,  hidden_channels)
    self.dur_emb = nn.Linear(1, hidden_channels)
    nn.init.normal_(self.x_emb.weight, 0.0, hidden_channels**-0.5)
    nn.init.normal_(self.ph_w_idx_emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = attentions.Encoder_VITS2(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout,
      gin_channels=self.gin_channels)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, w_dur_ms=None, ph_w_idx=None,  g=None):
    x = self.x_emb(x) / math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]

    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    if w_dur_ms is not None:
      w_dur_ms = self.dur_emb(w_dur_ms.unsqueeze(2)).permute(0,2,1) / math.sqrt(self.hidden_channels)
      x = x + w_dur_ms
    if ph_w_idx is not None:
      ph_w_idx = torch.diff(ph_w_idx, dim=1, prepend=ph_w_idx.new_zeros(x.size(0), 1)) >= 0
      ph_w_idx = self.ph_w_idx_emb(ph_w_idx.long()).permute(0,2,1) / math.sqrt(self.hidden_channels)  # [B, T_ph, H]
      x = x + ph_w_idx
    
    enc_in = x * x_mask 
    x = self.encoder(enc_in, x_mask, g=g)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask

class TextEncoder(nn.Module):
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)
    self.emb = nn.Embedding(n_vocab, hidden_channels)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)

  def forward(self, x, x_lengths, noise_scale=1):
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    # Process attention
    x = self.encoder(x * x_mask, x_mask)
    x = self.proj(x) * x_mask
    m, logs = torch.split(x, self.out_channels, dim=1)
    x = (m + torch.randn_like(m) * torch.exp(logs) * noise_scale) * x_mask
    return x, x_mask

def swish(x):
    return x * torch.sigmoid(x)

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = nn.functional.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = nn.functional.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x) 
            x = nn.functional.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        #periods = [2,3,5,7,11] 

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


if __name__ == "__main__":

    from utils import get_hparams
    hps_path = "./configs/config.yaml"
    hparams = get_hparams(hps_path)

