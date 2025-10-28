from ddpm.utils.config import CONFIG
from ddpm.utils.generate import generate

from matplotlib import pyplot as plt 
from tqdm import tqdm
import numpy as np
import pandas as pd

cfg = CONFIG

generated_imgs = []

for i in tqdm(range(cfg.num_img_to_generate)):
    x_t = generate(cfg)
    x_t = 255 * x_t[0][0].numpy()
    generated_imgs.append(x_t.astype(np.uint8).flatten())

generated_df = pd.DataFrame(generated_imgs, columns=[f"pixel{i}" for i in range(784)])
generated_df.to_csv(cfg.generated_csv_path, index=False)

fig, axes = plt.subplots(8, 8, figsize=(5,5))

for i, ax in enumerate(axes.flat):
    ax.imshow(np.reshape(generated_imgs[i], (28,28)), cmap="gray")
    ax.axis("off")

plt.tight_layout()
plt.show()
