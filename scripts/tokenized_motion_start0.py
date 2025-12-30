# This script generates motion tokens that are strictly temporally aligned with detailed text.
# In 'tokenized_motion.py', a random starting index (idx) is chosen to slice the motion sequence.
# In this script, the motion sequence is always sliced starting from index 0.

import os
import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import numpy as np
import models.vqvae as vqvae
from dataloader.tokenizer_loader import DATALoader_start0
from options import option

args = option.get_args_parser()
args = args.parse_args()

args.vq_dir = "./dataset/HumanML3D/VQVAE_start0"
os.makedirs(args.vq_dir, exist_ok = True)

token_loader = DATALoader_start0(args.dataname, 1, unit_length=2**args.down_t)

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)

vqvae_pth = f"./checkpoints/pretrained_vqvae/{args.dataname}.pth"
print ('loading checkpoint from {}'.format(vqvae_pth))
ckpt = torch.load(vqvae_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

for batch in token_loader:
    pose, name = batch
    bs = pose.shape[0]

    pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
    target = net.encode(pose)
    target = target.cpu().numpy()


    np.save(os.path.join(args.vq_dir, name[0] +'.npy'), target[0])
