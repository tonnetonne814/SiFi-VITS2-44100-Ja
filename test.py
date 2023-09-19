import torch, os
from mel_processing import spectrogram_torch,spec_to_mel_torch
from utils import load_wav_to_torch
import utils
import numpy as np
import matplotlib.pyplot as plt
 
def main2():
    a = torch.zeros(size=(2,3))
    b = torch.zeros(size=(2,3))
    c = a*b
    print(c.shape)

def main():
    basepath = "./dataset_SVS/00000"
    spec, wav = get_audio(basepath + ".wav")
    mel = spec_to_mel_torch(
          spec.float(), 
          2048, 
          80,  
          44100, 
          0.0,  
          None)

    spec_img = plot_spectrogram_to_numpy(spec.data.cpu().numpy(),  "spec.png")
    mel_img = plot_spectrogram_to_numpy (mel.data.cpu().numpy() , "mel.png")

    f0 = torch.load(basepath + "_f0.pt")
    generate_graph(f0, "f0.png")
    return 0

def generate_graph(vector, path, 
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
    plt.savefig(path)
    plt.clf()
    plt.close()
    return image_numpy

def plot_spectrogram_to_numpy(spectrogram, path):
  import matplotlib
  import matplotlib.pylab as plt
  import numpy as np
  
  fig, ax = plt.subplots(figsize=(14,6))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.savefig(path)
  plt.close()
  return data

def get_audio(filename):
    #print(filename)
    audio_norm, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != 44100:
        raise ValueError("{} {} SR doesn't match target {} SR".format(
            sampling_rate, 44100))
    audio_norm = audio_norm.unsqueeze(0)
    spec_filename = filename.replace(".wav", "_spec.pt")
    if os.path.exists(spec_filename):
        spec = torch.load(spec_filename)
    else:
        spec = spectrogram_torch(audio_norm.float(), 2048,
            44100, 512, 2048,
            center=False).float()
        spec = torch.squeeze(spec, 0)
        if spec.size(1) == -1:
            print("ERROR SPEC")
        torch.save(spec, spec_filename)
    return spec.float(), audio_norm.float()

if __name__ == "__main__":
    main2()