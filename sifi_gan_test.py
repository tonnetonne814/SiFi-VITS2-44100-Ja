import torch 
import yaml

def main3():
    from sifigan.models.generator import SiFiGANGenerator
    from sifigan.models.discriminator import UnivNetMultiResolutionMultiPeriodDiscriminator
    from sifigan.utils import dilated_factor        
    from sifigan.utils.features import SignalGenerator
    
    config_path = "./configs/sifigan.yaml"
    with open(config_path, mode="r", encoding="utf-8") as f:
      hps = yaml.safe_load(f)
    
    # model load
    model = SiFiGANGenerator(**hps)

    # data prep
    B = 2
    T = 97
    hop_size = 512 
    dense_factors= [0.5, 1, 4, 8, 16]   
    df_f0_type = "cf0"
    sr = 44100
    in_channels = 43 # features for recon

    out_size = (B, 1, int(hop_size*T)) # answer

    x = torch.rand(size=(B, 1, int(hop_size*T)), dtype=torch.float32) 
    c = torch.rand(size=(B, in_channels, T), dtype=torch.float32)
    f0 = torch.rand(size=(B, 1, T), dtype=torch.float32)
    g  = torch.zeros(size=(B,256, 1))
    cf0 = torch.rand(size=(B, 1, T), dtype=torch.float32)
    singlef0 = torch.rand(size=(T,), dtype=torch.float32)
    single_x = torch.rand(size=(int(hop_size*T),), dtype=torch.float32)
    single_c = torch.rand(size=(in_channels, T), dtype=torch.float32)

    single_vuv, single_cf0, flag = get_continuos_f0(singlef0)# from dataset config
    signal_types= ["sine"]              # from dataset config
    noise_amp= 0.003                    # from dataset config
    sine_amp= 0.1                    # Sine amplitude.# from dataset config

    signal_generator = SignalGenerator(
            sample_rate=sr,
            hop_size=hop_size,
            sine_amp=sine_amp,
            noise_amp=noise_amp,
            signal_types=signal_types,
        )
    single_x_check = signal_generator(cf0)

    upsample_scales= hps["upsample_scales"]
    prod_upsample_scales = np.cumprod(upsample_scales)
    df_sample_rates = [sr / hop_size * s for s in prod_upsample_scales]
    dfs = []
    for df, us in zip(dense_factors, prod_upsample_scales):
        dfs += [
            np.repeat(dilated_factor(single_cf0, sr, df), us)
            if df_f0_type == "cf0"
            else np.repeat(dilated_factor(single_cf0, sr, df), us)
        ]
    for i in range(len(dense_factors)):
        assert len(single_x) * df_sample_rates[i] == len(dfs[i]) * sr

    dfs_batch = [[] for _ in range(len(dense_factors))]
    for i in range(len(dense_factors)):
        dfs_batch[i] += [
            dfs[i].astype(np.float32).reshape(-1, 1)
        ]  # list = [(T', 1), ...]
    
    for i in range(len(dense_factors)):
        dfs_batch[i] = torch.FloatTensor(np.array(dfs_batch[i])).transpose(
                2, 1
            )  # (B, 1, T')
    Voice, Exicitation = model(x=x, c=c, d=dfs_batch, g=g)
    print(Voice.shape)
    
    config_path = "./configs/univnet.yaml"
    with open(config_path, mode="r", encoding="utf-8") as f:
      hps_d = yaml.safe_load(f)
    discriminator = UnivNetMultiResolutionMultiPeriodDiscriminator(**hps_d)
    out = discriminator(Voice)

    return 0


import copy
import numpy as np
from scipy.interpolate import interp1d
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


if __name__ == "__main__":
    main()