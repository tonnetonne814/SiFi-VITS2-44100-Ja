import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction


"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8

class MultinomialDiffusion(torch.nn.Module):
    def __init__(self, num_classes, timesteps=1000, loss_type='vb_stochastic', parametrization='x0'):
        super(MultinomialDiffusion, self).__init__()
        assert loss_type in ('vb_stochastic', 'vb_all')
        assert parametrization in ('x0', 'direct')

        if loss_type == 'vb_all':
            print('Computing the loss using the bound on _all_ timesteps.'
                  ' This is expensive both in terms of memory and computation.')

        self.num_classes = num_classes
        #self._denoise_fn = denoise_fn       # ノイズ予測器、実装はTransformerになってる。PE有
        self.loss_type = loss_type
        self.shape = None
        self.num_timesteps = timesteps
        self.parametrization = parametrization

        alphas = cosine_beta_schedule(timesteps)

        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        # Convert to float32 and register buffers.
        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())

        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count', torch.zeros(timesteps))


    # 学習時のロス計算コード xはIDs、出力はロス
    def preprocess(self, x, t_int, detach_mean=False):
        if self.loss_type == 'vb_stochastic':
            #x_start = x # x= text_IDs =[Batch, seq_len]
            
            # これは使わない。通常は線形のステップになる
            #t, pt = self.sample_time(b, device, 'importance') # t=学習用整数ステップ数、pt=0~1(importance or uniform)。

            log_x_start = index_to_log_onehot(x, self.num_classes) # x_start

            log_sample = self.q_sample(log_x_start=log_x_start, t=t_int)

            ###############
            log_true_prob = self.q_posterior(                   ## t=tのノイズ環境下での正解分布を出力
                log_x_start=log_x_start, log_x_t=log_sample, t=t_int)

            model_in = log_onehot_to_index(log_sample) # ここでlog_x_t(float)がx_t(int)に変換される。    
            return model_in, [log_sample, log_x_start, log_true_prob]

    def postprocess(self, model_out, parameters, t_int, t_float, detach_mean=False):

        log_sample, log_x_start, log_true_prob = parameters

        if self.loss_type == 'vb_stochastic':

            log_pred = F.log_softmax(model_out, dim=1)    # model_out = [Batch, class, seq_len]

            if self.parametrization == 'x0':
                log_model_prob = self.q_posterior(log_x_start=log_pred, log_x_t=log_sample, t=t_int)
            elif self.parametrization == 'direct':
                log_model_prob = log_pred
            else:
                raise ValueError

            if detach_mean:
                log_model_prob = log_model_prob.detach()

            kl = self.multinomial_kl(log_true_prob, log_model_prob)
            kl = sum_except_batch(kl)

            decoder_nll = -log_categorical(log_x_start, log_model_prob)
            decoder_nll = sum_except_batch(decoder_nll)

            mask = (t_int == torch.zeros_like(t_int)).float()
            kl = mask * decoder_nll + (1. - mask) * kl

            # 重みづけ有無判定用計算
            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t_int)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t_int, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t_int, src=torch.ones_like(Lt2))

            kl_prior = self.kl_prior(log_x_start)

            # 重みづけロス計算
            vb_loss = kl / t_float + kl_prior

            return torch.sum(vb_loss)   # マイナスを掛けて渡すのが数式的に正解だが、最終的には符号は＋でlossに加算するので、＋のまま

        elif self.loss_type == 'vb_all':
            # Expensive, dont do it ;).
            return -self.nll(model_out)
        else:
            raise ValueError()

    def q_posterior(self, log_x_start, log_x_t, t):
        # ベイズの定理によりqの事後確率を計算する
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # マルコフ連鎖より→となる。 q(xt | xt-1, x0) = q(xt | xt-1).

        t_minus_1 = t - 1
        # マイナス除去。最終的にデコーダーでは使わない。
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1) # log_probs=[Batch, class, seq_len]

        num_axes = (1,) * (len(log_x_start.size()) - 1) # Batch軸を無視した次元の数
        t_broadcast = t.view(-1, *num_axes) * torch.ones_like(log_x_start) # t_broadcast=[Batch, class, seq_len]
        log_EV_qxtmin_x0 = torch.where(t_broadcast == 0, log_x_start, log_EV_qxtmin_x0)

        # x_tmin1 ではない。これが一般的な公式の使い方らしい
        # 理解するのは簡単ではないが、事実こうなっている（マジで謎
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = \
            unnormed_logprobs \
            - torch.logsumexp(unnormed_logprobs, dim=1, keepdim=True)

        return log_EV_xtmin_given_xt_given_xstart


    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - np.log(self.num_classes)
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape) # cumprod(a,t,b)=累積積計算 tはint型
        log_1_min_cumprod_alpha = extract(self.log_1_min_cumprod_alpha, t, log_x_start.shape)

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes)      # log_probs=[Batch, class, seq_len]
        )

        return log_probs

    
    def predict_start(self, log_x_t, t):
        x_t = log_onehot_to_index(log_x_t) # ここでlog_x_t(float)がx_t(int)に変換される

        out = self._denoise_fn(t, x_t)  # ノイズ予測モデル本体。

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out, dim=1)    # [Batch, class, seq_len]
        return log_pred

    def p_pred(self, log_x, t):
        if self.parametrization == 'x0':
            log_x_recon = self.predict_start(log_x, t=t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x, t=t)
        else:
            raise ValueError
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, log_x, t):
        model_log_prob = self.p_pred(log_x=log_x, t=t)
        out = self.log_sample_categorical(model_log_prob)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.log_alpha.device

        b = shape[0]
        # start with random normal image.
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def _sample(self, image_size, batch_size = 16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    # log(textID)から、q(xt|x0)の分散期待値を計算する　log (E[V])
    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def nll(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array)

            loss += kl

        loss += self.kl_prior(log_x_start)

        return loss

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(self, log_x_start, log_x_t, t, detach_mean=False):
        log_true_prob = self.q_posterior(                   ## t=tのノイズ環境下での正解を出力
            log_x_start=log_x_start, log_x_t=log_x_t, t=t)

        log_model_prob = self.p_pred(log_x=log_x_t, t=t)    ## t=tでの正解を予測

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    # タイムステップ作成　uniform=線形的
    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        elif method == 'uniform':
            t= torch.randint(self.num_timesteps, size=(b, 1, 1)).cuda()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def log_prob(self, x):
        b, device = x.size(0), x.device
        if self.training:
            return self._train_loss(x)

        else: # 評価用
            log_x_start = index_to_log_onehot(x, self.num_classes)

            t, pt = self.sample_time(b, device, 'importance')

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t)

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior

            return -loss

    # サンプリング開始時の初期log_zを生成する。
    @torch.no_grad()
    def generate_log_z(self, size):
        device = "cuda"
        uniform_logits = torch.zeros(size=size, device=device)
        log_z = self.log_sample_categorical(uniform_logits)
        return log_z

    # for文最初のやつ
    @torch.no_grad()
    def sample_preprocess(self, log_z, t, num_samples=1):
        device = "cuda"
        

        print(f'Sample timestep {t:4d}', end='\r')
        t = torch.full((num_samples,), t, device=device, dtype=torch.long)
        model_in = log_onehot_to_index(log_z) # ここでlog_x_t(float)がx_t(int)に変換される
        return model_in, log_z

    # モデルの出力を入れて、for文内の最後を出し、これがまたfor文最初にループする。
    @torch.no_grad()
    def sample_postprocess(self, model_out, log_z, t_int):
        t_int = torch.tensor(t_int, dtype=torch.int64, device="cuda")
        t_int = torch.unsqueeze(t_int, dim=0)
        model_log_prob = F.log_softmax(model_out, dim=1)    # [Batch, class, seq_len]

        if self.parametrization == 'x0':
            model_log_prob = self.q_posterior(
                log_x_start=model_log_prob, log_x_t=log_z, t=t_int)
        elif self.parametrization == 'direct':
            pass
        else:
            raise ValueError
        
        log_z = self.log_sample_categorical(model_log_prob)

        return log_z

    @torch.no_grad()
    def decode_log_z(self, log_z):
        return log_onehot_to_index(log_z) 

    @torch.no_grad()
    def p_sample(self, log_x, t):
        model_log_prob = self.p_pred(log_x=log_x, t=t)
        out = self.log_sample_categorical(model_log_prob)
        return out

    def log_sample_categorical(self, logits):
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample
    
    def sample_chain(self, num_samples):
        b = num_samples
        device = self.log_alpha.device
        uniform_logits = torch.zeros(
            (b, self.num_classes) + self.shape, device=device)

        zs = torch.zeros((self.num_timesteps, b) + self.shape).long()

        log_z = self.log_sample_categorical(uniform_logits)
        for i in reversed(range(0, self.num_timesteps)):
            print(f'Chain timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=device, dtype=torch.long)
            log_z = self.p_sample(log_z, t)

            zs[i] = log_onehot_to_index(log_z)
        print()
        return zs


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes) # x_onehot = [Batch, seq_len, classes ]

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order) # x_onehot = [Batch, classes, seq_len ]

    log_x = torch.log(x_onehot.float().clamp(min=1e-30)) # logを取る

    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas

