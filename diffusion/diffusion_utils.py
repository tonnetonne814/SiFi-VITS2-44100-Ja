import os
import numpy as np
import torch
import torch.nn as nn
import copy
import math 
from tqdm import tqdm

def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def rescale(x):
    """
    Rescale a tensor to 0-1
    """

    return (x - x.min()) / (x.max() - x.min())


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:]  == '.pkl':
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    #print(path, epoch, flush=True)
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return torch.normal(0, 1, size=size).cuda()


def calc_noise_scale_embedding(noise_scales, noise_scale_embed_dim_in):
    """
    Embed a noise scale $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    noise_scales (torch.long tensor, shape=(batchsize, 1)):     
                                noise scales for batch data
    noise_scale_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete noise scales
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, noise_scale_embed_dim_in)):
    """

    assert noise_scale_embed_dim_in % 2 == 0

    half_dim = noise_scale_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = noise_scales * _embed
    noise_scale_embed = torch.cat((torch.sin(_embed), 
                                      torch.cos(_embed)), 1)
    
    return noise_scale_embed


def calc_diffusion_hyperparams_given_beta(beta):
    """
    Compute diffusion process hyperparameters

    Parameters:
    beta (tensor):  beta schedule 
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), beta/alpha/sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    T = len(beta)
    alpha = 1 - beta
    sigma = beta + 0
    for t in range(1, T):
        alpha[t] *= alpha[t-1]  # \alpha^2_t = \prod_{s=1}^t (1-\beta_s)
        sigma[t] *= (1-alpha[t-1]) / (1-alpha[t])  # \sigma^2_t = \beta_t * (1-\alpha_{t-1}) / (1-\alpha_t)
    alpha = torch.sqrt(alpha)
    sigma = torch.sqrt(sigma)
    
    _dh = {}
    _dh["T"], _dh["beta"], _dh["alpha"], _dh["sigma"] = T, beta, alpha, sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def calc_diffusion_hyperparams(T, beta_0, beta_T, tau, N, beta_N, alpha_N, rho):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of noise scales
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), beta/alpha/sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    beta = torch.linspace(beta_0, beta_T, T)
    alpha = 1 - beta
    sigma = beta + 0
    for t in range(1, T):
        alpha[t] *= alpha[t-1]  # \alpha^2_t = \prod_{s=1}^t (1-\beta_s)
        sigma[t] *= (1-alpha[t-1]) / (1-alpha[t])  # \sigma^2_t = \beta_t * (1-\alpha_{t-1}) / (1-\alpha_t)
    alpha = torch.sqrt(alpha)
    sigma = torch.sqrt(sigma)
    
    _dh = {}
    _dh["T"], _dh["beta"], _dh["alpha"], _dh["sigma"] = T, beta, alpha, sigma
    _dh["tau"], _dh["N"], _dh["betaN"], _dh["alphaN"], _dh["rho"] = tau, N, beta_N, alpha_N, rho
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def sampling_given_noise_schedule(
        net,
        size,
        diffusion_hyperparams,
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

    IDs_gt, IDs_gt_len, dur_gt, speaker_ID = condition
    _dh = diffusion_hyperparams
    T, alpha = _dh["T"], _dh["alpha"]
    assert len(alpha) == T
    assert len(size) == 3

    N = len(inference_noise_schedule)
    beta_infer = inference_noise_schedule
    alpha_infer = 1 - beta_infer
    sigma_infer = beta_infer + 0
    for n in range(1, N):
        alpha_infer[n] *= alpha_infer[n - 1]
        sigma_infer[n] *= (1 - alpha_infer[n - 1]) / (1 - alpha_infer[n])
    alpha_infer = torch.sqrt(alpha_infer)
    sigma_infer = torch.sqrt(sigma_infer)

    # Mapping noise scales to time steps
    steps_infer = []
    for n in tqdm(range(N), desc="Mapping noise scales"):
        step = map_noise_scale_to_time_step(alpha_infer[n], alpha)
        if step >= 0:
            steps_infer.append(step)
    #print(steps_infer, flush=True)
    steps_infer = torch.FloatTensor(steps_infer)

    # N may change since alpha_infer can be out of the range of alpha
    N = len(steps_infer)

    print('begin sampling, total number of reverse steps = %s' % N)

    x = std_normal(size).cuda()
    log_z = net.multinomial_diff.generate_log_z(size)

    if return_sequence:
        x_ = copy.deepcopy(x)
        xs = [x_]

    f0_length = torch.tensor([dur_gt.size(1)], dtype=torch.int64, device="cuda:0")
    with torch.no_grad():
        for n in range(N - 1, -1, -1):
            diffusion_steps = (steps_infer[n] * torch.ones((size[0], 1))).cuda()

            # preprocess
            model_in, log_z = net.multinomial_diff.sample_preprocess(log_z=log_z, t=n)

            # 出力も要修正
            epsilon_theta, model_out =  net(f0=x,  
                                f0_len=f0_length,
                                IDs=IDs_gt, 
                                IDs_len=IDs_gt_len, 
                                vuv=model_in,                  
                                vuv_len=f0_length,               
                                attn=dur_gt, 
                                timesteps=diffusion_steps,
                                g=speaker_ID,
                                sampling=True)
            
            # postprocess
            log_z = net.multinomial_diff.sample_postprocess(model_out=model_out, log_z=log_z, t_int=n)

            if ddim:
                alpha_next = alpha_infer[n] / (1 - beta_infer[n]).sqrt()
                c1 = alpha_next / alpha_infer[n]
                c2 = -(1 - alpha_infer[n] ** 2.).sqrt() * c1
                c3 = (1 - alpha_next ** 2.).sqrt()
                x = c1 * x + c2 * epsilon_theta + c3 * epsilon_theta  # std_normal(size)
            else:
                x -= beta_infer[n] / torch.sqrt(1 - alpha_infer[n] ** 2.) * epsilon_theta
                x /= torch.sqrt(1 - beta_infer[n])
                if n > 0:
                    x = x + sigma_infer[n] * std_normal(size)
            if return_sequence:
                x_ = copy.deepcopy(x)
                xs.append(x_)
    if return_sequence:
        return xs
    return x, net.multinomial_diff.decode_log_z(log_z)

def noise_scheduling(net, size, diffusion_hyperparams, condition=None, ddim=False):
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
    noise schedule:                 a list of noise scales in torch.tensor, length <= N
    """

    _dh = diffusion_hyperparams
    N, betaN, alphaN, rho, alpha = _dh["N"], _dh["betaN"], _dh["alphaN"], _dh["rho"], _dh["alpha"]

    print('begin noise scheduling, maximum number of reverse steps = %d' % (N))

    betas = []
    x = std_normal(size)
    with torch.no_grad():
        beta_cur = torch.ones(1, 1, 1).cuda() * betaN
        alpha_cur = torch.ones(1, 1, 1).cuda() * alphaN
        for n in range(N - 1, -1, -1):
            # print(n, beta_cur.squeeze().item(), alpha_cur.squeeze().item())
            step = map_noise_scale_to_time_step(alpha_cur.squeeze().item(), alpha)
            if step >= 0:
                betas.append(beta_cur.squeeze().item())
            diffusion_steps = (step * torch.ones((size[0], 1))).cuda()
            epsilon_theta = net((x, condition, diffusion_steps,))
            if ddim:
                alpha_nxt = alpha_cur / (1 - beta_cur).sqrt()
                c1 = alpha_nxt / alpha_cur
                c2 = -(1 - alpha_cur ** 2.).sqrt() * c1
                c3 = (1 - alpha_nxt ** 2.).sqrt()
                x = c1 * x + c2 * epsilon_theta + c3 * epsilon_theta  # std_normal(size)
            else:
                x -= beta_cur / torch.sqrt(1 - alpha_cur ** 2.) * epsilon_theta
                x /= torch.sqrt(1 - beta_cur)
            alpha_nxt, beta_nxt = alpha_cur, beta_cur
            alpha_cur = alpha_nxt / (1 - beta_nxt).sqrt()
            if alpha_cur > 1:
                break
            beta_cur = net.noise_pred(
                x.squeeze(1), (beta_nxt.view(-1, 1), (1 - alpha_cur ** 2.).view(-1, 1)))
            if beta_cur.squeeze().item() < rho:
                break
    return torch.FloatTensor(betas[::-1]).cuda()


def theta_timestep_loss(net, X, diffusion_hyperparams, reverse=False):
    """
    Compute the training loss for learning theta

    Parameters:
    net (torch network):            the wavenet models
    X (tuple, shape=(2,)):          training data in tuple form (mel_spectrograms, audios)
                                    mel_spectrograms: torch.tensor, shape is batchsize followed by each mel_spectrogram shape
                                    audios: torch.tensor, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    theta loss
    """
    # assert type(X) == tuple and len(X) == 2

    ##############
    # ロス関数定義# 
    loss_fn = nn.MSELoss()
    ##############
    
    f0, f0_lengths, dur, IDs, IDs_lengths, vuv, vuv_lengths, speakerID = X
    _dh = diffusion_hyperparams
    T, alpha = _dh["T"], _dh["alpha"]

    B, C, L = f0.shape  
    #ts = torch.randint(T, size=(B, 1, 1)).cuda()  # ランダムなステップを生成 t=1~T → cudaへ # F0 Diffusionのもの。
    ts, pt = net.multinomial_diff.sample_time(B, device="cuda", method='importance')
    noise_gt = std_normal(f0.shape)                  # ノイズ生成
    ts=ts.view(B, 1, 1)
    
    # q(x_t|x_0)からx_tを計算
    delta = (1 - alpha[ts] ** 2.).sqrt() 
    alpha_cur = alpha[ts]
    noisy_f0 = alpha_cur * f0 + delta * noise_gt  

    # ノイズを予測する。
    f0_noise_pd, vuv_loss_minus = net(f0=noisy_f0,              # f0→noizy_f0へ修正済み。
                              f0_len=f0_lengths,
                              IDs=IDs, 
                              IDs_len=IDs_lengths, 
                              vuv=vuv,                  # 用修正、本来はnoisy_vuv ?
                              vuv_len=vuv_lengths,
                              attn=dur,
                              g=speakerID,
                              pt=pt,
                              timesteps=ts.view(B, 1), 
                              sampling=False)
    
    if reverse:
        x0 = (noisy_f0 - delta * f0_noise_pd) / alpha_cur
        return loss_fn(f0_noise_pd, noise_gt), x0
    
    num_elem = torch.sum(vuv_lengths)
    vuv_loss_minus = vuv_loss_minus.sum() / (math.log(2) * num_elem)
    loss_f0 = loss_fn(f0_noise_pd, noise_gt)
    return loss_f0 - vuv_loss_minus


def phi_loss(net, X, diffusion_hyperparams):
    """
    Compute the training loss for learning phi
    Parameters:
    net (torch network):            the wavenet models
    X (tuple, shape=(2,)):          training data in tuple form (mel_spectrograms, audios)
                                    mel_spectrograms: torch.tensor, shape is batchsize followed by each mel_spectrogram shape
                                    audios: torch.tensor, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    phi loss
    """
    assert type(X) == tuple and len(X) == 2
    _dh = diffusion_hyperparams
    T, alpha, tau = _dh["T"], _dh["alpha"], _dh["tau"]

    mel_spectrogram, audio = X
    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    ts = torch.randint(tau, T - tau, size=(B,)).cuda()  # randomly sample steps from 1~T
    alpha_cur = alpha.index_select(0, ts).view(B, 1, 1)
    alpha_nxt = alpha.index_select(0, ts + tau).view(B, 1, 1)
    beta_nxt = 1 - (alpha_nxt / alpha_cur) ** 2.
    delta = (1 - alpha_cur ** 2.).sqrt()
    z = std_normal(audio.shape)
    noisy_audio = alpha_cur * audio + delta * z  # compute x_t from q(x_t|x_0)
    epsilon_theta = net((noisy_audio, mel_spectrogram, ts.view(B, 1),))
    beta_est = net.noise_pred(noisy_audio.squeeze(1), (beta_nxt.view(B, 1), delta.view(B, 1) ** 2.))
    phi_loss = 1 / (2. * (delta ** 2. - beta_est)) * (
            delta * z - beta_est / delta * epsilon_theta) ** 2.
    phi_loss += torch.log(1e-8 + delta ** 2. / (beta_est + 1e-8)) / 4.
    phi_loss = (torch.mean(phi_loss, -1, keepdim=True) + beta_est / delta ** 2 / 2.).mean()

    return phi_loss


def compute_hyperparams_given_schedule(beta):
    """
    Compute diffusion process hyperparameters

    Parameters:
    beta (tensor):  beta schedule

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), beta/alpha/sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    T = len(beta)
    alpha = 1 - beta
    sigma = beta + 0
    for t in range(1, T):
        alpha[t] *= alpha[t - 1]  # \alpha^2_t = \prod_{s=1}^t (1-\beta_s)
        sigma[t] *= (1 - alpha[t - 1]) / (1 - alpha[t])  # \sigma^2_t = \beta_t * (1-\alpha_{t-1}) / (1-\alpha_t)
    alpha = torch.sqrt(alpha)
    sigma = torch.sqrt(sigma)

    _dh = {}
    _dh["T"], _dh["beta"], _dh["alpha"], _dh["sigma"] = T, beta, alpha, sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams



def map_noise_scale_to_time_step(alpha_infer, alpha):
    if alpha_infer < alpha[-1]:
        return len(alpha) - 1
    if alpha_infer > alpha[0]:
        return 0
    for t in range(len(alpha) - 1):
        if alpha[t+1] <= alpha_infer <= alpha[t]:
             step_diff = alpha[t] - alpha_infer
             step_diff /= alpha[t] - alpha[t+1]
             return t + step_diff.item()
    return -1


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):
                                dimensionality of the embedding space for discrete diffusion steps

    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).cuda()
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed),
                                      torch.cos(_embed)), 1)

    return diffusion_step_embed

if __name__ == "__main__":
    noise_gt = std_normal((1,1,800))
    print(torch.max(noise_gt))
    print(torch.min(noise_gt))