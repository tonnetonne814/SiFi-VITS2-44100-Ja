import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
import commons

######### export TORCH_CUDNN_V8_API_DISABLED=1
torch.cuda.empty_cache()
import torch._dynamo.config
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True

from models import VITS2_based_SiFiSinger
from sifigan.models.discriminator import UnivNetMultiResolutionMultiPeriodDiscriminator
from models import DurationDiscriminator

from sifigan.losses.reg import ResidualLoss

from losses import generator_loss, discriminator_loss, feature_loss,kl_loss
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import utils
from data_utils import TextAudioLoader_SVS_synth, TextAudioCollate_SVS_synth, DistributedBucketSampler

# torch.autograd.set_detect_anomaly(True) # 逆誤差伝搬のエラー部分を自動で見つける機能
torch.backends.cudnn.benchmark = True # 学習早くなるらしい（知らんけど
global_step = 0

import warnings
warnings.filterwarnings(action='ignore')

def main():

  """ GPUのみの学習を許可 """
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '65520'

  hps = utils.get_hparams()   # config.yamlやargparseは、ここで読み込まれている。

  if hps["DDP_use"] is True:
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))  # DDP ON(未確認)
  else:
    run(rank=0, n_gpus=1, hps=hps)                     # DDP OFF

def run(rank, n_gpus, hps):
  global global_step
  if rank == 0: # 1番目のGPUであれば入る（GPUが複数ある場合を想定
    logger = utils.get_logger(hps["model_dir"])
    logger.info(hps)
    # utils.check_git_hash(hps["model_dir"])
    writer = SummaryWriter(log_dir=hps["model_dir"])
    writer_eval = SummaryWriter(log_dir=os.path.join(hps["model_dir"], "eval"))

  # Linux Only
  if hps["DDP_use"] is True: 
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)   

  # 乱数の再現性確保
  torch.manual_seed(hps["seed"])
  torch.cuda.set_device(rank)

  # データセットの読み込み君 
  train_dataset = TextAudioLoader_SVS_synth(audiopaths_and_text= hps["train_data_path"] ,
                                  hparams            = hps)
  vocab_size = train_dataset.get_ph_vocab_size()
  
  # 似た長さのデータをモデルに流す君
  train_sampler = DistributedBucketSampler(dataset     = train_dataset,
                                           batch_size  = hps["batch_size"],
                                           boundaries  = [32,300,400,500,600,700,800,900,1000],
                                           num_replicas= n_gpus,
                                           rank        = rank,
                                           shuffle     = True) # シャッフル指定

  # 入力するデータに0を加えて長さを統一する君
  collate_fn = TextAudioCollate_SVS_synth(hparams=hps)

  # 上三つを統べる
  train_loader = DataLoader(dataset=train_dataset, 
                            num_workers=hps["num_workers"],     # データ読み込みに使用するCPUのスレッド数
                            shuffle=False, 
                            pin_memory=True,    # Trueだとなんか動作早くなるらしい（知らんけど
                            collate_fn=collate_fn, 
                            batch_sampler=train_sampler)

  # 1番目のGPUであれば
  if rank == 0:
    eval_dataset = TextAudioLoader_SVS_synth(audiopaths_and_text=hps["eval_data_path"], 
                                   hparams            =hps)

    eval_loader = DataLoader(dataset      = eval_dataset, 
                             num_workers  = 1, 
                             shuffle      = False,
                             batch_size   = hps["batch_size"], 
                             pin_memory   = True,
                             drop_last    = False, 
                             collate_fn   = collate_fn)

  net_dur_disc = DurationDiscriminator(
    hps["dur_discriminator"]["hidden_channels"], 
    hps["dur_discriminator"]["hidden_channels"], 
    3, 
    0.1, 
    gin_channels=hps["common"]["gin_channels"] if hps["common"]["n_speaker"] != 0 else 0,
    ).cuda(rank)

  net_g = VITS2_based_SiFiSinger(
      hps = hps,
      n_vocab=vocab_size,
      spec_channels=hps["spec_encoder"]["spec_channels"],
      segment_size=int(hps["segments_size"] // hps["hop_length"]),
      n_speakers=hps["common"]["n_speaker"],
      gin_channels=hps["common"]["gin_channels"],
      **hps["hifi_gan"]).cuda(rank)


  net_d = UnivNetMultiResolutionMultiPeriodDiscriminator(**hps["UnivNet_MRMPDiscriminator"]).cuda(rank)
  
  # torch.compile 学習高速化
  #net_g = torch.compile(net_g)
  #net_d = torch.compile(net_d)
  #if net_dur_disc is not None:
  #  net_dur_disc = torch.compile(net_dur_disc)
  
  # 最適化手法
  optim_g = torch.optim.AdamW(params      = net_g.parameters(),
                              lr          = float(hps['learning_rate']), 
                              betas       =       hps["betas"], 
                              eps         = float(hps["eps"]))
  optim_d = torch.optim.AdamW(params      = net_d.parameters(),
                              lr          = float(hps['learning_rate']), 
                              betas       =       hps["betas"], 
                              eps         = float(hps["eps"]))
  if net_dur_disc is not None:
    optim_dur_disc = torch.optim.AdamW(params      = net_dur_disc.parameters(),
                                       lr          = float(hps['learning_rate']), 
                                       betas       =       hps["betas"], 
                                       eps         = float(hps["eps"]))
  else:
    optim_dur_disc = None
 
  # checkpoint 確認
  in_progress_checkpoint_path = utils.latest_checkpoint_path(hps["model_dir"], "G_*.pth")
  if os.path.isfile(in_progress_checkpoint_path): # 学習再開
    try:
      _, _, _, epoch_str = utils.load_checkpoint(in_progress_checkpoint_path, net_g, optim_g)
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps["model_dir"], "D_*.pth"), net_d, optim_d)
      if net_dur_disc is not None:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps["model_dir"], "DUR_*.pth"), net_dur_disc, optim_dur_disc)
      global_step = (epoch_str - 1) * len(train_loader)
      print("### Resume learning mode ###")
    except:
      net_g = utils.load_model_diffsize(in_progress_checkpoint_path, net_g, hps, optim_g)
      net_d = utils.load_model_diffsize(utils.latest_checkpoint_path(hps["model_dir"], "D_*.pth"), net_d, hps, optim_d)
      if net_dur_disc is not None:
        net_dur_disc = utils.load_model_diffsize(utils.latest_checkpoint_path(hps["model_dir"], "DUR_*.pth"), net_dur_disc, hps, optim_dur_disc)
      epoch_str = 1
      global_step = 0.
      print("### Resume learning mode ###")
  elif hps["is_finetune"] is True: # ファインチューニング
    net_g = utils.load_model_diffsize(hps["finetune_G"], net_g, hps, optim_g)
    net_d = utils.load_model_diffsize(hps["finetune_D"], net_d, hps, optim_d)
    if net_dur_disc is not None:
      net_dur_disc = utils.load_model_diffsize(hps["finetune_DUR"], net_dur_disc, hps, optim_dur_disc)
    epoch_str = 1
    global_step = 0
    print("### FineTuning mode ###")
  else: # どちらでもない→初期状態から
    epoch_str = 1
    global_step = 0
    print("### Initial state mode ###")
  global_step = (epoch_str - 1) * len(train_loader)

  # エポックに応じて自動で学習率を変える
  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer  = optim_g, 
                                                       gamma      = hps["lr_decay"], 
                                                       last_epoch = epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer  = optim_d, 
                                                       gamma      = hps["lr_decay"], 
                                                       last_epoch = epoch_str-2)
  if net_dur_disc is not None:
    scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(optimizer  = optim_dur_disc, 
                                                       gamma      = hps["lr_decay"], 
                                                       last_epoch = epoch_str-2)
  else:
    scheduler_dur_disc = None

  # float16で計算して，高速化するか
  scaler = GradScaler(enabled=hps["fp16_run"])

  # Linux Only
  if hps["DDP_use"] is True: 
    net_g = DDP(net_g, device_ids=[rank]) 
    net_d = DDP(net_d, device_ids=[rank]) 
    if net_dur_disc is not None:
      net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)

  # 学習/評価処理
  for epoch in range(epoch_str, int(hps["max_epochs"]) + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d, net_dur_disc], [optim_g, optim_d, optim_dur_disc], [scheduler_g, scheduler_d, scheduler_dur_disc], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d, net_dur_disc], [optim_g, optim_d, optim_dur_disc], [scheduler_g, scheduler_d, scheduler_dur_disc], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()
    if net_dur_disc is not None:
      scheduler_dur_disc.step()

  # 最終モデルの保存
  utils.save_checkpoint(net_g, optim_g, hps["learning_rate"], epoch, os.path.join(hps["model_dir"], "G_LastModel.pth".format(global_step)))
  utils.save_checkpoint(net_d, optim_d, hps["learning_rate"], epoch, os.path.join(hps["model_dir"], "D_LastModel.pth".format(global_step)))
  if net_dur_disc is not None:
    utils.save_checkpoint(net_dur_disc, optim_dur_disc, hps["learning_rate"], epoch, os.path.join(hps["model_dir"], "DUR_LastModel.pth".format(global_step)))

def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  net_g, net_d, net_dur_disc = nets
  optim_g, optim_d, optim_dur_disc = optims
  scheduler_g, scheduler_d, scheduler_dur_disc = schedulers
  
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step
  
  reg_loss = ResidualLoss(sample_rate=hps["sampling_rate"],
                          fft_size=hps["filter_length"],
                          hop_size=hps["hop_length"],
                          f0_floor=hps["f0_min"],
                          f0_ceil=hps["f0_max"],
                          n_mels=hps["n_mel_channels"]).cuda()

  net_g.train() # 学習モードへ
  net_d.train() # 学習モードへ
  if net_dur_disc is not None:
    net_dur_disc.train()
  for batch_idx, (wav,\
                  spec,            spec_lengths,\
                  f0,\
                  ph_IDs,          ph_IDs_lengths,\
                  ph_frame_dur,\
                  word_dur_ms,     \
                  word_frame_dur,  word_frame_dur_lengths,\
                  ph_idx_in_a_word,  \
                  n_ph_pooling,\
                  dfs, \
                  Sinewaves, speakerID) in enumerate(train_loader):

    # Forward Process
    with autocast(enabled=hps["fp16_run"]):
      
      # y_hat is sliced 
      y_hat, excitation, l2_dur_loss, attn_gt, ids_slice, ph_IDs_mask, spec_mask, \
      (z, z_p, m_p, logs_p, m_q, logs_q), \
      (hidden_x, logw, logw_) = net_g(spec=spec.cuda(rank),
                                              spec_lengths=spec_lengths.cuda(rank),
                                              ph_IDs=ph_IDs.cuda(rank), 
                                              ph_IDs_lengths=ph_IDs_lengths.cuda(rank),
                                              ph_dur=ph_frame_dur.cuda(rank), 
                                              word_frame_dur = word_frame_dur.cuda(rank),
                                              word_frame_dur_lenngths=word_frame_dur_lengths.cuda(rank),
                                              word_dur_ms = word_dur_ms.cuda(rank),
                                              ph_word_flag=ph_idx_in_a_word.cuda(rank),
                                              n_ph_pool=n_ph_pooling.cuda(rank),
                                              dfs=dfs,  # slice後にcudaに乗せる
                                              sinewave=Sinewaves,
                                              speakerID=speakerID.cuda(rank))
      mel = spec_to_mel_torch(
          spec.cuda(rank), 
          hps["filter_length"], 
          hps["n_mel_channels"],  
          hps["sampling_rate"], 
          hps["mel_fmin"],  
          hps["mel_fmax"])

      # slice process
      y = commons.slice_segments(wav.cuda(rank), ids_slice * hps["hop_length"], hps["segments_size"])
      y_mel = commons.slice_segments(mel, ids_slice, hps["segments_size"] // hps["hop_length"] )
      f0_slice = commons.slice_segments(f0.cuda(rank), ids_slice , hps["segments_size"]//hps["hop_length"])

      y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1),
                                        hps["filter_length"],
                                        hps["n_mel_channels"],
                                        hps["sampling_rate"],
                                        hps["hop_length"],
                                        hps["win_length"],
                                        hps["mel_fmin"],
                                        hps["mel_fmax"])
      
      ### Duration Discriminator ###
      if net_dur_disc is not None:  
        y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x.detach(), ph_IDs_mask.detach(), logw.detach(), logw_.detach())
        with autocast(enabled=False):
          # TODO: I think need to mean using the mask, but for now, just mean all
          loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
          loss_dur_disc_all = loss_dur_disc
        optim_dur_disc.zero_grad()
        scaler.scale(loss_dur_disc_all).backward()
        scaler.unscale_(optim_dur_disc)
        grad_norm_dur_disc = commons.clip_grad_value_(net_dur_disc.parameters(), None)
        scaler.step(optim_dur_disc)

      ### Discriminator ### 
      y_d_hat_g = net_d(y_hat.detach()) 
      y_d_hat_r = net_d(y)
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc    

    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d) 

    ### Generator ###
    with autocast(enabled=hps["fp16_run"]):
      y_d_hat_g, fmap_g = net_d(y_hat, return_fmaps=True)  # fake feature
      with torch.no_grad():
        _, fmap_r = net_d(y, return_fmaps=True) # real feature

      if net_dur_disc is not None:
        y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, ph_IDs_mask, logw, logw_)
      with autocast(enabled=False):
        loss_ph_dur    = torch.sum(l2_dur_loss[0]).float() * hps["gain_dur_ph"]
        loss_word_dur  = torch.sum(l2_dur_loss[1]).float() * hps["gain_dur_word"]
        loss_length    = torch.sum(l2_dur_loss[2]).float() * hps["gain_dur_length"]
        loss_length_l2 = loss_ph_dur + loss_word_dur + loss_length
        loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * hps["gain_mel"]
        loss_kl_spec = kl_loss(z_p, logs_q, m_p, logs_p, spec_mask) * hps["gain_kl_spec"]
        loss_fm = feature_loss(fmap_r, fmap_g) * hps["gain_fm"]
        loss_gen, losses_gen = generator_loss(y_d_hat_g) 
        loss_reg = reg_loss(s=excitation, y=y, f=f0_slice) * hps["gain_reg"] # sifi gan
        ############################################################
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_length_l2 + loss_kl_spec + loss_reg#+ loss_f0
        ############################################################
        if net_dur_disc is not None:
          loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
          loss_gen_all += loss_dur_gen

    losses = loss_gen_all
    optim_g.zero_grad()
    scaler.scale(losses).backward()   
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.unscale_(optim_g)
    scaler.step(optim_g)
    scaler.update()

    # 記録 / 評価
    if rank==0:
      if global_step % hps["log_interval"] == 0:
        lr = optim_g.param_groups[0]['lr']
        #losses = 
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info("steps:{}   [Generator]   ALL:{:.5f} | mel:{:.5f} | fm:{:.5f} | reg:{:.5f}| kl:{:.5f} | dur:{:.5f}"
                    .format(global_step, loss_gen_all.item(), loss_mel.item(), loss_fm.item(),loss_reg.item(), loss_kl_spec.item(),loss_length_l2.item()
                    ))
        logger.info("steps:{} [Discriminator] d_ALL:{:.5f} | d_dur_ALL:{:.5f} | "
                    .format(global_step, loss_disc_all.item(), loss_dur_disc_all.item()
                    ))
        
        scalar_dict = {"loss/g/ALL": loss_gen_all, 
                       "loss/d/ALL": loss_disc_all,
                       "loss/g/AdvDiscDur" :loss_dur_gen,
                       "loss/g/AdvDisc" :loss_gen,
                       "loss/g/Regularization" : loss_reg,
                       "loss/g/KL_div_spec" : loss_kl_spec,
                       "loss/g/FeatureMatching": loss_fm, 
                       "loss/g/MelReconstruction": loss_mel, 
                       "loss/g/Duration_SUM": loss_length_l2, 
                       "loss/g/Duration_ph": loss_ph_dur, 
                       "loss/g/Duration_word": loss_word_dur, 
                       "loss/g/Duration_length": loss_length, 
                       "learning_rate": lr, 
                       "grad_norm/discriminator": grad_norm_d, 
                       "grad_norm/generator": grad_norm_g}
        
        scalar_dict.update({"loss/g/Adversarial_Deceive{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d/Adversarial_Real_{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d/Adversarial_Fake_{}".format(i): v for i, v in enumerate(losses_disc_g)})
        
        if net_dur_disc is not None:
          scalar_dict.update({"loss/d_dur/ALL"    : loss_dur_disc_all,})
          scalar_dict.update({"grad_norm/duration_discriminator": grad_norm_dur_disc})
          scalar_dict.update({"loss/d_dur/Adversarial_Real_{}".format(i): v for i, v in enumerate(losses_dur_disc_r)})
          scalar_dict.update({"loss/d_dur/Adversarial_Fake_{}".format(i): v for i, v in enumerate(losses_dur_disc_g)})

        # attnマップの作成
        length_scale=1
        w = torch.exp(logw) * ph_IDs_mask * length_scale
        w_ceil = torch.ceil(w)
        spec_mask = torch.unsqueeze(commons.sequence_mask(spec_lengths, spec.size(2)), 1).to(spec.dtype) # [B, 1, ph_len]
        ph_IDs_mask = torch.unsqueeze(commons.sequence_mask(ph_IDs_lengths, ph_IDs.size(1)), 1).to(ph_IDs.dtype) # [B, 1, ph_len]
        attn_mask     = torch.unsqueeze(spec_mask, 2) * torch.unsqueeze(ph_IDs_mask, -1)    # attn_mask = [B, 1, ph_len, note(word)_len]
        attn_pd       = commons.generate_path(duration=torch.unsqueeze(w_ceil,dim=1), mask=attn_mask.permute(0,1,3,2).cuda(rank))

        image_dict = { 
            "slice/mel_orgiginal": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_generated": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel_full_length": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn_pd": utils.plot_alignment_to_numpy(attn_pd[0][0].data.cpu().numpy()),
            "all/attn_gt": utils.plot_alignment_to_numpy(attn_gt.permute(0,2,1)[0].data.cpu().numpy()),
            "all/attn_overlap": utils.plot_alignment_to_numpy(0.5*(attn_pd[0][0].data.cpu().numpy()) + 0.5*attn_gt.permute(0,2,1)[0].data.cpu().numpy())
        }

        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps["eval_interval"] == 0:
        evaluate(hps=hps,
                 epoch=epoch,
                 nets=[net_g,net_d,net_dur_disc],
                 eval_loader=eval_loader, 
                 writer_eval=writer_eval)

        utils.save_checkpoint(model=net_g, 
                              optimizer=optim_g, 
                              learning_rate=hps["learning_rate"], 
                              iteration=epoch, 
                              checkpoint_path=os.path.join(hps["model_dir"], "G_{}.pth".format(global_step)))
        utils.save_checkpoint(model=net_d, 
                              optimizer=optim_d, 
                              learning_rate=hps["learning_rate"], 
                              iteration=epoch, 
                              checkpoint_path=os.path.join(hps["model_dir"], "D_{}.pth".format(global_step)))
        if net_dur_disc is not None:
          utils.save_checkpoint(model=net_dur_disc, 
                              optimizer=optim_dur_disc, 
                              learning_rate=hps["learning_rate"], 
                              iteration=epoch, 
                              checkpoint_path=os.path.join(hps["model_dir"], "DUR_{}.pth".format(global_step)))
    global_step += 1
    #torch.cuda.empty_cache()

  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))
  
def evaluate(hps, epoch, nets, eval_loader, writer_eval):
    net_g,net_d,net_dur_disc = nets

    net_g.eval()
    net_d.eval()
    if net_dur_disc is not None:
      net_d.eval()

    reg_loss = ResidualLoss(sample_rate=hps["sampling_rate"],
                          fft_size=hps["filter_length"],
                          hop_size=hps["hop_length"],
                          f0_floor=hps["f0_min"],
                          f0_ceil=hps["f0_max"],
                          n_mels=hps["n_mel_channels"]).cuda()
    
    rank = 0
    
    count = 0
    loss_gen_all = 0
    loss_gen_adv_all = 0
    loss_disc_all= 0
    loss_dur_gen_all= 0
    #loss_f0_all= 0
    loss_reg_all = 0
    loss_kl_spec_all= 0
    loss_fm_all= 0
    loss_mel_all= 0
    loss_length_l2_all= 0
    loss_dur_ph_all = 0
    loss_dur_word_all = 0
    loss_dur_length_all = 0
    loss_dur_disc_all = 0
    with torch.no_grad():
      for batch_idx, (wav,\
                      spec,            spec_lengths,\
                      f0,\
                      ph_IDs,          ph_IDs_lengths,\
                      ph_frame_dur,\
                      word_dur_ms,     \
                      word_frame_dur,  word_frame_dur_lengths,\
                      ph_idx_in_a_word,  \
                      n_ph_pooling,\
                      dfs, \
                      Sinewaves, speakerID) in enumerate(eval_loader):
        
        count += 1

        # Forward Process
        with autocast(enabled=hps["fp16_run"]):

          # y_hat is sliced 
          y_hat, excitation, l2_dur_loss, attn_gt, ids_slice, ph_IDs_mask, spec_mask, \
          (z, z_p, m_p, logs_p, m_q, logs_q), \
          (hidden_x, logw, logw_) = net_g(spec=spec.cuda(rank),
                                              spec_lengths=spec_lengths.cuda(rank),
                                              ph_IDs=ph_IDs.cuda(rank), 
                                              ph_IDs_lengths=ph_IDs_lengths.cuda(rank),
                                              ph_dur=ph_frame_dur.cuda(rank), 
                                              word_frame_dur = word_frame_dur.cuda(rank),
                                              word_frame_dur_lenngths=word_frame_dur_lengths.cuda(rank),
                                              word_dur_ms = word_dur_ms.cuda(rank),
                                              ph_word_flag=ph_idx_in_a_word.cuda(rank),
                                              n_ph_pool=n_ph_pooling.cuda(rank),
                                              dfs=dfs,  # slice後にcudaに乗せる
                                              sinewave=Sinewaves,
                                              speakerID=speakerID.cuda(rank))

          mel = spec_to_mel_torch(
              spec.cuda(rank), 
              hps["filter_length"], 
              hps["n_mel_channels"],  
              hps["sampling_rate"], 
              hps["mel_fmin"],  
              hps["mel_fmax"])

          # sliced 
          y_mel = commons.slice_segments(mel, ids_slice, hps["segments_size"] // hps["hop_length"] )
          f0_slice = commons.slice_segments(f0.cuda(rank), ids_slice , hps["segments_size"]//hps["hop_length"])

          y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1),
                                            hps["filter_length"],
                                            hps["n_mel_channels"],
                                            hps["sampling_rate"],
                                            hps["hop_length"],
                                            hps["win_length"],
                                            hps["mel_fmin"],
                                            hps["mel_fmax"])

          y = commons.slice_segments(wav.cuda(rank), ids_slice * hps["hop_length"], hps["segments_size"])

          ### Duration Discriminator ###
          if net_dur_disc is not None:  
            y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x.detach(), ph_IDs_mask.detach(), logw.detach(), logw_.detach())
            with autocast(enabled=False):
              # TODO: I think need to mean using the mask, but for now, just mean all
              loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
              loss_dur_disc_all += loss_dur_disc
              
          ### Discriminator ### 
          y_d_hat_g = net_d(y_hat.detach()) 
          y_d_hat_r = net_d(y)
          with autocast(enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all += loss_disc    

        ### Generator ###
        with autocast(enabled=hps["fp16_run"]):

          #y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

          y_d_hat_g, fmap_g = net_d(y_hat, return_fmaps=True)  # fake feature
          with torch.no_grad():
            _, fmap_r = net_d(y, return_fmaps=True) # real feature

          if net_dur_disc is not None:
            y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, ph_IDs_mask, logw, logw_)
          with autocast(enabled=False):
          
            loss_ph_dur    = torch.sum(l2_dur_loss[0]).float()  * hps["gain_dur_ph"]
            loss_word_dur  = torch.sum(l2_dur_loss[1]).float()  * hps["gain_dur_word"]
            loss_length    = torch.sum(l2_dur_loss[2]).float()  * hps["gain_dur_length"]
            loss_length_l2 = loss_ph_dur + loss_word_dur + loss_length
            loss_dur_ph_all += loss_ph_dur
            loss_dur_word_all += loss_word_dur
            loss_dur_length_all += loss_length
            loss_length_l2_all += loss_length_l2

            loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * hps["gain_mel"]
            loss_mel_all += loss_mel

            loss_kl_spec = kl_loss(z_p, logs_q, m_p, logs_p, spec_mask) * hps["gain_kl_spec"]
            loss_kl_spec_all += loss_kl_spec
            
            loss_fm = feature_loss(fmap_r, fmap_g) * hps["gain_fm"]
            loss_fm_all += loss_fm

            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_adv_all += loss_gen 

            loss_reg = reg_loss(s=excitation, y=y, f = f0_slice) * hps["gain_reg"]
            loss_reg_all += loss_reg

            if net_dur_disc is not None:
              loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
              loss_dur_gen_all += loss_dur_gen

      loss_gen_all = loss_gen + loss_fm + loss_mel + loss_length_l2 + loss_kl_spec + loss_reg#+ loss_f0
            
      scalar_dict = {"loss/g/ALL"               : loss_gen_all / count, 
                     "loss/d/ALL"               : loss_disc_all/ count,
                     "loss/g/AdvDiscDur"        : loss_dur_gen_all/ count,
                     "loss/g/AdvDisc"           : loss_gen_adv_all/ count,
                     "loss/g/Regularization"    : loss_reg_all/ count,
                     "loss/g/KL_div_spec"       : loss_kl_spec_all/ count,
                     "loss/g/FeatureMatching"   : loss_fm_all/ count, 
                     "loss/g/MelReconstruction" : loss_mel_all/ count, 
                     "loss/g/Duration_SUM": loss_length_l2_all/ count, 
                     "loss/g/Duration_ph": loss_dur_ph_all/ count, 
                     "loss/g/Duration_word": loss_dur_word_all/ count, 
                     "loss/g/Duration_length": loss_dur_length_all/ count
                     }
      if net_dur_disc is not None:
        scalar_dict.update({"loss/d_dur/ALL"    : loss_dur_disc_all})

    wav_gt                 = wav[0].view(1, 1,-1)
    ph_IDs                 = ph_IDs[0].view(1,-1).cuda(rank)                
    ph_IDs_len             = ph_IDs_lengths[0].view(-1).cuda(rank)       
    ph_IDs_dur             = ph_frame_dur[0].view(1,-1).cuda(rank)  
    f0_gt                  = f0  [0][0].view(1,1,-1).cuda(rank)
    f0_gt_len              = torch.tensor(torch.sum(ph_IDs_dur), dtype=torch.int64).view(1).cuda(rank)
    word_frame_dur         = word_frame_dur[0].view(1,-1).cuda(rank)  
    speakerID              = speakerID[0].view(-1).cuda(rank)      
    word_frame_dur_lengths = word_frame_dur_lengths[0].view(-1)
    word_dur_ms            = word_dur_ms[0].view(1,-1)
    ph_idx_in_a_word       = ph_idx_in_a_word[0].view(1,-1)
    n_ph_pooling           = n_ph_pooling[0].view(1,-1)
    Sinewaves              = Sinewaves[0].view(1,1,-1)
    for idx, df in enumerate(dfs):
      dfs[idx] = df[0].view(1,1,-1)

    with torch.no_grad():
      # with f0_gt
      y_hat_with_f0gt, attn, y_mask = net_g.eval_infer(ph_IDs=ph_IDs.cuda(rank), 
                                                  ph_IDs_lengths=ph_IDs_len.cuda(rank),
                                                  word_frame_dur = word_frame_dur.cuda(rank),
                                                  word_frame_dur_lenngths=word_frame_dur_lengths.cuda(rank),
                                                  word_dur_ms = word_dur_ms.cuda(rank),
                                                  ph_word_flag=ph_idx_in_a_word.cuda(rank),
                                                  n_ph_pool=n_ph_pooling.cuda(rank),
                                                  speakerID=speakerID.cuda(rank),
                                                  dfs=dfs, 
                                                  sinewave=Sinewaves
                                                  ) 

    try:
      y_hat_with_f0gt = y_hat_with_f0gt[0].to('cpu').detach().numpy().copy()
    except: 
      y_hat_with_f0gt = None

    f0_mask = torch.unsqueeze(commons.sequence_mask(f0_gt_len, f0_gt.size(2)), 1).to(f0_gt.dtype) # [B, 1, ph_len]
    ph_IDs_mask = torch.unsqueeze(commons.sequence_mask(ph_IDs_len, ph_IDs.size(1)), 1).to(ph_IDs.dtype) # [B, 1, ph_len]
    attn_mask     = torch.unsqueeze(f0_mask, 2) * torch.unsqueeze(ph_IDs_mask, -1)    # attn_mask = [B, 1, ph_len, note(word)_len]
    attn_gt       = commons.generate_path(duration=torch.unsqueeze(ph_IDs_dur,dim=1), mask=attn_mask.permute(0,1,3,2))

    image_dict = { 
    "all/attn": utils.plot_alignment_to_numpy(attn.permute(0,2,1)[0].data.cpu().numpy()),
    "all/attn_GT": utils.plot_alignment_to_numpy(attn_gt[0][0].data.cpu().numpy()),
    "all/attn_overlap": utils.plot_alignment_to_numpy(0.5*(attn_gt[0][0].data.cpu().numpy()) + 0.5*attn.permute(0,2,1)[0].data.cpu().numpy())
    }
    
    audio_dict = {f"GroundTruth/audio_all" : wav_gt}
    if y_hat_with_f0gt is not None:
      audio_dict.update({f"gen/audio_with_f0gt_{batch_idx}"        : y_hat_with_f0gt})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      scalars=scalar_dict,
      images=image_dict,
      audios = audio_dict,
      audio_sampling_rate=hps["sampling_rate"]
    )

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:
      net_dur_disc.train()

def generate_graph(vector, 
                   label="NoName",
                   color="blue", 
                   title="Title",
                   x_label = 'Frames',
                   y_label = "y_labels",
                   figsize=(36,8)):
   
    fig = plt.figure(figsize=figsize) 
    x = np.arange(0, len(vector))
    plt.plot(x, vector, label=label, color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.ylim(y_val_min, y_val_max)
    fig.canvas.draw() 
    plot_image = fig.canvas.renderer._renderer  
    image_numpy = np.array(plot_image)
    plt.clf()
    plt.close()
    return image_numpy

def moving_average(loss,n=5):

    # Build a tensor list
    loss_list = []

    if len(loss_list) >= n:
        # drop the first loss
        loss_list =  loss_list[1:,:]
  
    loss_list.append(loss.data)

    baseline = loss-sum(loss_list)/len(loss_list)
    
    if baseline.all() == 0:
        return loss.detach()
    else:
        return baseline


# このファイルを実行したら、ココから始まる                  
if __name__ == "__main__":
  os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"
  main()
