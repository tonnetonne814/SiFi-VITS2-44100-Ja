import torch.nn as nn
from diffusion.diffusion_utils import  compute_hyperparams_given_schedule, std_normal, map_noise_scale_to_time_step
import  torch
from tqdm import tqdm

class F0_Diffusion(nn.Module):
    def __init__(self, hps ):
        super(F0_Diffusion, self).__init__()

        self.hps = hps
        self.diffusion_hyperparameters = self.make_diffusion_hyperparams()
        self.alpha = self.diffusion_hyperparameters["alpha"]

        # ロス関数定義はここ
        self.loss_fn = nn.MSELoss()

    def get_diff_hyperparameters(self):
        return self.diffusion_hyperparameters
    
    def make_diffusion_hyperparams(self, noise_schedule="cosine_beta", training=True):
        
        beta = self.get_beta(noise_schedule=noise_schedule,training=training)

        diffusion_hyperparams = compute_hyperparams_given_schedule(beta)
        for key in diffusion_hyperparams:
            if key in ["beta", "alpha", "sigma"]:
                diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()
        return diffusion_hyperparams
    
    def get_beta(self, noise_schedule, timesteps=None, training=True):
        
        if timesteps is None:   
            timesteps = int(self.hps["Diffusion"]["T"])

        if training is True:
            beta_0 = 0.0001
            beta_T = 0.9999
        else:
            beta_0 = float(self.hps["Diffusion"]["beta_0"])
            beta_T = float(self.hps["Diffusion"]["beta_T"])
                       
        if noise_schedule == "cosine_beta":
            beta = self.get_cosine_beta_schedule(beta_start=beta_0, beta_end=beta_T, timesteps=timesteps)
        elif noise_schedule == "quadratic_beta":
            beta = self.get_quadratic_beta_schedule(beta_start=beta_0, beta_end=beta_T, timesteps=timesteps)
        elif noise_schedule == "sigmoid_beta":
            beta = self.get_sigmoid_beta_schedule(beta_start=beta_0, beta_end=beta_T, timesteps=timesteps)
        else: # Liner schedule
            noise_schedule = "linear"
            beta = torch.linspace(beta_0, beta_T,timesteps).cuda()
        
        print(f"# Noise Schedule # : {noise_schedule}  beta_0:{beta_0}, beta_T:{beta_T}")

        return beta

    def get_cosine_beta_schedule(self, beta_start, beta_end, timesteps, s=0.008): # timesteps = 1000
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, beta_start, beta_end)

    def get_quadratic_beta_schedule(self, beta_start, beta_end, timesteps):
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

    def get_sigmoid_beta_schedule(self, beta_start, beta_end, timesteps):
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    def get_noisy_f0(self, f0, ts):
        noise_gt = std_normal(f0.shape)                  
        ts=ts.view(f0.size(0), 1, 1)
        self.delta = (1 - self.alpha[ts] ** 2.).sqrt() 
        self.alpha_cur = self.alpha[ts]
        noisy_f0 = self.alpha_cur * f0 + self.delta * noise_gt  # q(x_t|x_0)
        return noisy_f0, noise_gt
    
    def get_loss(self, f0_noise_pd, noisy_f0, noise_gt , reverse=False):

        if reverse:
            x0 = (noisy_f0 - self.delta * f0_noise_pd) / self.alpha_cur
            return self.loss_fn(f0_noise_pd, noise_gt), x0

        loss_f0 = self.loss_fn(f0_noise_pd, noise_gt)
        return loss_f0
    
    @torch.no_grad()
    def get_noise_schedule(self, timesteps=None, training=False):
        
        # Specify a specific noise schedule
        if self.hps["Diffusion"]['noise_schedule'] != '':
            noise_schedule = self.hps["Diffusion"]['noise_schedule']
            if isinstance(noise_schedule, list):
                beta = torch.FloatTensor(noise_schedule).cuda()
            else:
                beta = self.get_beta(noise_schedule=noise_schedule, timesteps=timesteps, training=training)

        # Select Schedule
        else:
            try:
                reverse_step = int(self.hps["Diffusion"]['N'])
            except:
                print('Please specify $N (the number of revere iterations) in config file. Now denoise with 4 iterations.')
                reverse_step = 4
            if reverse_step == 1000:
                beta = torch.linspace(0.000001, 0.01, 1000).cuda()
            elif reverse_step == 200:
                beta = torch.linspace(0.0001, 0.02, 200).cuda()
            # Below are schedules derived by Noise Predictor.
            elif reverse_step == 8:
                beta = [6.689325005027058e-07, 1.0033881153503899e-05, 0.00015496854030061513,
                                 0.002387222135439515, 0.035597629845142365, 0.3681158423423767, 0.4735414385795593, 0.5]
            elif reverse_step == 6:
                beta = [1.7838445955931093e-06, 2.7984189728158526e-05, 0.00043231004383414984,
                                0.006634317338466644, 0.09357017278671265, 0.6000000238418579]
            elif reverse_step == 4:
                beta = [3.2176e-04, 2.5743e-03, 2.5376e-02, 7.0414e-01]
            elif reverse_step == 3:
                beta = [9.0000e-05, 9.0000e-03, 6.0000e-01]
            else:
                raise NotImplementedError

        # put schedule on cuda
        #if isinstance(noise_schedule, list):
        #  noise_schedule = torch.FloatTensor(noise_schedule).cuda()

        return torch.FloatTensor(beta).cuda()
    
    @torch.no_grad()
    def generate_x(self, size):
        return std_normal(size)
    
    def get_step_infer(self, size, inference_noise_schedule):
        self.size=size
        T, alpha = self.diffusion_hyperparameters["T"], self.diffusion_hyperparameters["alpha"]
        assert len(alpha) == T
        assert len(size) == 3

        N = len(inference_noise_schedule)
        self.beta_infer = inference_noise_schedule
        alpha_infer = 1 - self.beta_infer
        sigma_infer = self.beta_infer + 0
        for n in range(1, N):
            alpha_infer[n] *= alpha_infer[n - 1]
            sigma_infer[n] *= (1 - alpha_infer[n - 1]) / (1 - alpha_infer[n])
        self.alpha_infer = torch.sqrt(alpha_infer) # self
        self.sigma_infer = torch.sqrt(sigma_infer) # self

        # Mapping noise scales to time steps
        steps_infer = []
        for n in tqdm(range(N), desc="Mapping noise scales"):
            step = map_noise_scale_to_time_step(self.alpha_infer[n], alpha)
            if step >= 0:
                steps_infer.append(step)
        #print(steps_infer, flush=True)
        steps_infer = torch.FloatTensor(steps_infer)

        # N may change since alpha_infer can be out of the range of alpha
        N = len(steps_infer)

        return N, steps_infer
    
    @torch.no_grad()
    def denoising(self, x, noise_pd, ddim, n):
        
        if ddim:
            alpha_next = self.alpha_infer[n] / (1 - self.beta_infer[n]).sqrt()
            c1 = alpha_next / self.alpha_infer[n]
            c2 = -(1 - self.alpha_infer[n] ** 2.).sqrt() * c1
            c3 = (1 - alpha_next ** 2.).sqrt()
            x = c1 * x + c2 * noise_pd + c3 * noise_pd  # std_normal(size)
        else:
            x -= self.beta_infer[n] / torch.sqrt(1 - self.alpha_infer[n] ** 2.) * noise_pd
            x /= torch.sqrt(1 - self.beta_infer[n])
            if n > 0:
                x = x + self.sigma_infer[n] * self.generate_x(self.size)
            else:
                #print()
                pass
        return x
    