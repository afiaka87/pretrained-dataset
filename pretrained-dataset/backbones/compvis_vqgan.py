from base import EncoderBackbone, ImageGlobDataset
import os
import torch

from dalle_pytorch.vae import VQGanVAE

comp_vis_vqgan_encoder_configs = {
    'imagenet_vqgan': {
        'ckpt_url': 'http://batbot.tv/ai/models/VQGAN/imagenet_1024_slim.ckpt',
        'config_url': 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
    },
    'imagenet_vqgan_large': {
        'ckpt_url': 'http://batbot.tv/ai/models/VQGAN/imagenet_16384_slim.ckpt',
        'config_url': 'http://batbot.tv/ai/models/VQGAN/imagenet_16384.yaml',
    },
    'coco_vqgan': {
        'ckpt_url': 'http://batbot.tv/ai/models/VQGAN/coco_first_stage.ckpt',
        'config_url': 'http://batbot.tv/ai/models/VQGAN/coco_first_stage.yaml'
    },
    'gumbel_vqgan': {
        'ckpt_url': 'http://batbot.tv/ai/models/VQGAN/gumbel_f8_8192.ckpt',
        'config_url': 'https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
    },
}


class CompVisVqgan(EncoderBackbone):
    def __init__(self, cfg=comp_vis_vqgan_encoder_configs['imagenet_vqgan'], device='cpu', cache_dir=os.path.expanduser('~/.cache/pretrained-dataset')):
        self.cfg = cfg
        self.ckpt_url = cfg['ckpt_url']
        self.config_url = cfg['config_url']
        model = VQGanVAE(self.config_url, self.ckpt_url)
        model.eval().requires_grad_(False)
        self.vqgan_model = model.to(device)
        super().__init__(cfg)

    def encode(self, x):
        image_fmap_size = (self.vqgan_model.image_size //
                           (2 ** self.vqgan_model.num_layers))
        image_seq_len = image_fmap_size ** 2
        num_img_tokens = int(0.4375 * image_seq_len)
        return self.vqgan_model.get_codebook_indices(x)[:, :num_img_tokens]

    def encode_all_images(self, image_glob: str, output_dir: str, device: torch.device, batch_size: int, shuffle: bool = True, num_workers: int = 0):
        dataset = ImageGlobDataset(
            image_glob, image_size=self.vqgan_model.image_size)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        if len(dataloader) == 0:
            raise ValueError("No images found")
        for idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            encoded_images = self.vqgan_model.encode(batch)
            yield torch.save(encoded_images, os.path.join(output_dir, f"{idx}.pt"))

    def decode(self, x):
        return self.vqgan_model.decode(x) # TODO write tests for this and encode