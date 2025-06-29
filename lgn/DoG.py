import torch
import numpy as np
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from torch.nn.functional import conv2d
from scipy.signal import convolve2d


class DOG_kernel:
  def __init__(self, dim, ang, ppa, ctr, sur):
    """
    Creates a dogKernel object representing a kernel with passed overall, centre, and surround dimensions.

    :param dim: Kernel dimensions in pixels (size of the kernel object).
    :param ang: Kernel dimensions in angular units (used for scaling or angular representation).
    :param ctr: Size of the center within the kernel (in pixels).
    :param sur: Size of the surround within the kernel (in pixels).
    :object kernel: The actual kernel itself, an np.array set with set_filter_coefficients
    """
    self.dim = dim
    self.ang = ang
    self.ppa = ppa
    self.ctr = ctr
    self.sur = sur
    self.kernel = None

  def __str__(self):
    return f"___________________________\nKernel:\n{self.kernel}\nKernel Size: {np.shape(self.kernel)}\nKernel Center: {self.ctr}\nSurround Center: {self.sur}"

  def set_filter_coefficients(self, ONOFF):
    """
    Sets the filter coefficents of the dogKernel using the dogKernel's attributes (dim, ang, ctr, sur)
    :param self:
    :param ONOFF: ON sets the kernel with a center-on/surround-off distribution, OFF sets the kernel with a center-off/surround-on distribution
    :return self.kernel: Sets the filter coefficients of the dogKernel with a difference of gaussian distribution
    """

    ctr_kernel = self.gen_gaussian_kernel(self.dim, self.ctr)
    sur_kernel = self.gen_gaussian_kernel(self.dim, self.sur)

    if ONOFF == 'ON':
      self.kernel = +ctr_kernel - sur_kernel
    elif ONOFF == 'OFF':
      self.kernel = -ctr_kernel + sur_kernel
    else:
      print(f"Incorrect argument given, passed {ONOFF} to ONOFF, when it should be 'ON' or 'OFF'")

    self.kernel = self.kernel - np.mean(self.kernel)

    input_min = 0
    input_max = 1

    self.kernel = self.kernel / (np.sum(np.abs(self.kernel)) / 2) * (input_max - input_min)

  def gen_gaussian_kernel(self, shape, sigma):
    """
    Creates a 2D Gaussian kernel.
    :param shape: size of the kernel (np.array)
    :param sigma: standard deviation for the Gaussian
    :return: Gaussian kernel (np.array)
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x*x + y*y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    h /= h.sum()  # Normalize to sum to 1
    return h
  
  def display_kernel(self, show_plt=True, show_hist=True, show_img=True):
    """
    Kernel display function
    :param plt: toggle pyplot (bool)
    :param hist: toggle histogram (bool)
    :param img: toggle example img, Fashion-MNIST boot (bool)
    """
    display = plt.figure()

    if show_plt == True:
      display.add_subplot(1,3,1)
      plt.imshow(self.kernel, cmap='grey')
      plt.title("Kernel")

    if show_hist == True:
      display.add_subplot(1,3,2)
      plt.hist(self.kernel.flatten(), bins=20, range=[np.min(self.kernel), np.max(self.kernel)])
      plt.title("Histogram")
    
    if show_img == True:
      display.add_subplot(1,3,3)
      test_img = np.load('test_img.npy')
      test_img = np.clip(convolve2d(test_img, self.kernel, mode='same'), 0, None)
      plt.imshow(test_img, cmap='grey')
      plt.title("Image")


    plt.show()


class DualDoG:
    """
    PyTorch transform that takes a single-channel image [1xHxW] and
    returns a two-channel tensor [2xHxW] where
      channel 0 = ON-convolution,
      channel 1 = OFF-convolution.
    """
    def __init__(self, on_kernel: DOG_kernel, off_kernel: DOG_kernel):
        # build [1×1×kH×kW] FloatTensors once
        self.on_w  = torch.from_numpy(on_kernel.kernel).unsqueeze(0).unsqueeze(0).float()
        self.off_w = torch.from_numpy(off_kernel.kernel).unsqueeze(0).unsqueeze(0).float()

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: [1×H×W], FloatTensor in [0,1]
        x = img.unsqueeze(0)               # → [1×1×H×W]
        on  = F.conv2d(x, self.on_w,  padding='same')
        off = F.conv2d(x, self.off_w, padding='same')
        org = x
        # cat → [1×2×H×W], then squeeze batch → [2×H×W]
        return torch.cat([on, off, org], dim=1).squeeze(0)