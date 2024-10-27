#%%
class ChapmanPalette:

    def __init__(self) -> None:
        names = ['white', 'pantherBlack', 'chapmanRed', 'grove', 'pacific', 'pillar', 'sand', 'dk_blue', 'lt_blue']
        colors = [
            (255, 255, 255, 255),
            (35, 31, 32, 255),
            (165, 0, 52, 255),
            (0, 150, 108, 255),
            (0, 156, 166, 255),
            (110, 98, 89, 255),
            (221, 203, 164, 255),
            (25, 35, 45, 255),
            (175, 207, 255, 255)
        ]
        for c,n in zip(colors, names):
            self.__dict__[n] = c
        for c,n in zip([self.pillar, self.sand], ['pillar', 'sand']):
            for v in [25, 50, 75, ]:
                self.__dict__[f'{n}{v}'] = tuple([c[i]+int((255-c[i])*v/100) if i<3 else 255 for i in range(4) ])
        pass



def to_gpl(palette = None):
    if palette is None:
        palette = ChapmanPalette()
    with open('chapman.gpl', 'w') as f:
        f.write('GIMP Palette\n')
        f.write('Name: Chapman\n')
        f.write('#\n')
        for name, color in palette.__dict__.items():
            f.write('{:3} {:3} {:3} {}\n'.format(color[0], color[1], color[2], name))




if __name__ == '__main__':
    to_gpl()
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

f = '/Users/cgirard/Library/CloudStorage/OneDrive-ChapmanUniversity/template_dark/ppt/media/image1.png'
p = ChapmanPalette()

from_colors = np.array([p.white, (0,0,0,255), p.chapmanRed, p.grove, p.pacific, p.pillar, p.sand,])/255
to_colors = np.array([p.dk_blue, p.sand50, p.chapmanRed, p.grove, p.pacific, p.pillar, p.sand,])/255
orig = np.asarray(Image.open(f),dtype=float)/255
plt.imshow(orig)

# %%
dest = np.zeros_like(orig, dtype=float)
ar_shape = [n for n in orig.shape]
ar_shape[-1]=len(from_colors)
o_indices =  np.empty(ar_shape)

for i, c in enumerate(from_colors):
    d = np.linalg.norm(orig- np.array(c[:3]), axis = -1)
    # d= np.dot(orig, np.array(c[:3]))
    o_indices[:,:,i] = d

o_arrray = np.array(o_indices)
norms = np.sum(1-o_arrray, axis=-1, keepdims=True)
scales = 1/norms
scales[norms<0.5] = 0
scales[scales>1] = 1

o_scaled = o_arrray * scales
# o_scaled = 1-o_arrray


im=plt.imshow(np.max(o_scaled, axis=-1))#, cmap= 'tab10')
plt.colorbar(im)


## %%
f,ax = plt.subplots()

for i, c in enumerate(to_colors):
    colors = np.zeros((2, 4))
    colors[0, :3] = to_colors[0, :3]
    colors[1, :3] = c[:3]
    colors[1,-1] = 1.
    cm = mcolors.LinearSegmentedColormap.from_list('test', colors)

    # mult = np.tile(c[:3], (ar_shape[0], ar_shape[1], 1))
    # dest = np.stack
    # f,ax = plt.subplots()

    ax.imshow(o_scaled[:,:,i], 
              cmap = cm)
            #   color = plt.colormaps
            #   alpha = o_scaled[:,:,i])

# plt.imshow(dest)
# %%
