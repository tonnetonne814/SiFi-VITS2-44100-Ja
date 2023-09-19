import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from models import DiffusionModels

######### export TORCH_CUDNN_V8_API_DISABLED=1
torch.cuda.empty_cache()
import torch._dynamo.config
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose=True

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import utils
from data_utils import TextAudioLoader_TTS_f0diff, TextAudioCollate_TTS_f0diff, DistributedBucketSampler

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
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))  # DDP ON
  else:
    run(rank=0, n_gpus=1, hps=hps)                     # DDP OFF


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0: # 1番目のGPUであれば入る（GPUが複数ある場合を想定
    logger = utils.get_logger(hps["model_dir"])
    logger.info(hps)
    writer = SummaryWriter(log_dir=hps["model_dir"])
    writer_eval = SummaryWriter(log_dir=os.path.join(hps["model_dir"], "eval"))

  # Linux Only
  if hps["DDP_use"] is True: 
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)   

  # 乱数の再現性確保
  torch.manual_seed(hps["seed"])
  torch.cuda.set_device(rank)

  # データセットの読み込み君 
  train_dataset = TextAudioLoader_TTS_f0diff(audiopaths_and_text= hps["train_data_path"] ,
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
  collate_fn = TextAudioCollate_TTS_f0diff()

  # 上三つを統べる
  train_loader = DataLoader(dataset=train_dataset, 
                            num_workers=hps["num_workers"],     # データ読み込みに使用するCPUのスレッド数
                            shuffle=False, 
                            pin_memory=True,    # Trueだとなんか動作早くなるらしい（知らんけど
                            collate_fn=collate_fn, 
                            batch_sampler=train_sampler)

  # 1番目のGPUであれば
  if rank == 0:
    eval_dataset = TextAudioLoader_TTS_f0diff(audiopaths_and_text=hps["eval_data_path"], 
                                              hparams            =hps)
    
    eval_loader = DataLoader(dataset      = eval_dataset, 
                             num_workers  = 1, 
                             shuffle      = False,
                             batch_size   = hps["batch_size"], 
                             pin_memory   = True,
                             drop_last    = False, 
                             collate_fn   = collate_fn)

  # モデルを作成
  net_g = DiffusionModels(hps=hps, vocab_size=vocab_size).cuda(rank)
  #net_g = torch.compile(net_g)

  # 最適化手法
  optim_g = torch.optim.AdamW(params      = net_g.parameters(),
                              lr          = float(hps['learning_rate']), 
                              betas       =       hps["betas"], 
                              eps         = float(hps["eps"]))

  # checkpoint 確認
  in_progress_checkpoint_path = utils.latest_checkpoint_path(hps["model_dir"], "G_*.pth")
  if os.path.isfile(in_progress_checkpoint_path): # 学習再開
    try:
      _, _, _, epoch_str = utils.load_checkpoint(in_progress_checkpoint_path, net_g, optim_g)
      global_step = (epoch_str - 1) * len(train_loader)
      print("### Resume learning mode ###")
    except:
      net_g = utils.load_model_diffsize(in_progress_checkpoint_path, net_g, hps, optim_g)
      epoch_str = 1
      global_step = 0.
      print("### Resume learning mode ###")
  elif hps["is_finetune"] is True: # ファインチューニング
    net_g = utils.load_model_diffsize(hps["finetune_G"], net_g, hps, optim_g)
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

  # float16で計算して，高速化するか
  scaler = GradScaler(enabled=hps["fp16_run"])

  # Linux Only
  if hps["DDP_use"] is True: 
    net_g = DDP(net_g, device_ids=[rank]) 

  # 学習/評価処理
  for epoch in range(epoch_str, int(hps["max_epochs"]) + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, net_g, optim_g, scheduler_g, scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, net_g, optim_g, scheduler_g, scaler, [train_loader, None], None, None)
    scheduler_g.step()

  # 最終モデルの保存
  utils.save_checkpoint(net_g, optim_g, hps["learning_rate"], epoch, os.path.join(hps["model_dir"], "G_LastModel.pth".format(global_step)))
    
def train_and_evaluate(rank, epoch, hps, net_g, optim_g, scheduler_g, scaler, loaders, logger, writers):
  
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train() # 学習モードへ
  for batch_idx, (f0,f0_lengths,\
                  ph_IDs,ph_IDs_lengths,\
                  ph_frame_dur,\
                  speakerID) in enumerate(train_loader):

    # Forward Process
    with autocast(enabled=hps["fp16_run"]):
      loss_f0 = net_g(f0=f0.cuda(rank),  
                      f0_len=f0_lengths.cuda(rank),
                      IDs=ph_IDs.cuda(rank), 
                      IDs_len=ph_IDs_lengths.cuda(rank),
                      IDs_dur=ph_frame_dur.cuda(rank),
                      g=speakerID.cuda(rank))
    #losses = loss_f0 + loss_vuv
    losses = loss_f0
    optim_g.zero_grad()
    scaler.scale(losses).backward()   
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
        #logger.info([x.item() for x in [losses, loss_f0, loss_vuv]] + [global_step, lr])
        logger.info([x.item() for x in [losses]] + [global_step, lr])
        #scalar_dict = {"loss": losses, "loss_f0" : loss_f0, "loss_vuv":loss_vuv}
        scalar_dict = {"loss_f0": losses}
        
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          scalars=scalar_dict)
        
      if global_step % hps["eval_interval"] == 0:
        evaluate(hps=hps,
                 net_g=net_g,
                 eval_loader=eval_loader, 
                 writer_eval=writer_eval)
        utils.save_checkpoint(model=net_g, 
                              optimizer=optim_g, 
                              learning_rate=hps["learning_rate"], 
                              iteration=epoch, 
                              checkpoint_path=os.path.join(hps["model_dir"], "G_{}.pth".format(global_step)))
    global_step += 1

  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))
  
def evaluate(hps, net_g, eval_loader, writer_eval):
    net_g.eval()
    rank = 0
    count = 0
    loss_f0_sum = 0
    with torch.no_grad():
      for batch_idx, (f0,f0_lengths,\
                      ph_IDs,ph_IDs_lengths,\
                      ph_frame_dur,\
                      speakerID) in enumerate(eval_loader):
        # Forward Process
        with autocast(enabled=hps["fp16_run"]):
          loss_f0 = net_g(f0=f0.cuda(rank),  
                          f0_len=f0_lengths.cuda(rank),
                          IDs=ph_IDs.cuda(rank), 
                          IDs_len=ph_IDs_lengths.cuda(rank),
                          IDs_dur=ph_frame_dur.cuda(rank),
                          g=speakerID.cuda(rank))
          loss_f0_sum += loss_f0
          count += 1

    f0_gt=f0                  [0][0].view(1, 1,-1).cuda(rank)
    f0_gt_len=f0_lengths      [0]   .view(-1)     .cuda(rank)
    ph_IDs=ph_IDs             [0]   .view(1,-1)   .cuda(rank)   
    ph_IDs_len=ph_IDs_lengths [0]   .view(-1)     .cuda(rank)
    ph_IDs_dur=ph_frame_dur   [0]   .view(1,-1)   .cuda(rank)
    speakerID=speakerID       [0]   .view(-1)     .cuda(rank)

    #f0_pd, vuv_pd = net_g.sampling(condition=[IDs_gt, IDs_gt_len, dur_gt, speaker_ID])
    f0_pd = net_g.sampling(condition=[ph_IDs, ph_IDs_len, ph_IDs_dur, 
                                      None, None, None, speakerID])

    scalar_dict = {"loss_f0": loss_f0_sum/count}
    #scalar_dict = {"loss": loss_f0_sum + loss_vuv_sum,
    #               "loss_f0": loss_f0_sum,
    #               "loss_vuv" : loss_vuv_sum}

    f0_gt = f0_gt[0][0].to('cpu').detach().numpy().copy()
    f0_pd = f0_pd[0][0].to('cpu').detach().numpy().copy() 
    #vuv_gt= vuv_gt.view(-1).to('cpu').detach().numpy().copy() 
    #vuv_pd= vuv_pd.view(-1).to('cpu').detach().numpy().copy() 

    f0_gt_image= generate_graph(vector=f0_gt, 
                                label="F0 gt",
                                color="red", 
                                x_label = 'Frames',
                                y_label = "f0")

    f0_pd_image= generate_graph(vector=f0_pd, 
                                label="F0 pd",
                                color="blue", 
                                x_label = 'Frames',
                                y_label = "f0")
    #vuv_gt_image=generate_graph(vector=vuv_gt, 
    #                            label="Voice Unvoice Value",
    #                            color="red", 
    #                            x_label = 'Frames',
    #                            y_label = "0=mask, 1=uv, 2=v")
    #vuv_pd_image= generate_graph(vector=vuv_pd, 
    #                            label="Voice Unvoice Value",
    #                            color="blue", 
    #                            x_label = 'Frames',
    #                            y_label = "0=mask, 1=uv, 2=v",)
    
    image_dict = { 
    "f0/gt":f0_gt_image ,
    "f0/pd":f0_pd_image ,
    #"vuv/gt":vuv_gt_image ,
    #"vuv/pd":vuv_pd_image ,
        }
    
    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      scalars=scalar_dict,
      images=image_dict
    )
    net_g.train()

def generate_graph(vector, 
                   label="NoName",
                   color="blue", 
                   title="Title",
                   x_label = 'Frames',
                   y_label = "y_labels",
                   figsize=(20,5)):
   
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

# このファイルを実行したら、ココから始まる                  
if __name__ == "__main__":
  os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"
  main()
