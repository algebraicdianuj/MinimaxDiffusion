"""
Sample new images from a pre-trained DiT.
"""
import os
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import time

class_dict = {'airplane': 'n04552348',
                'automobile': 'n04037443',
                'bird': 'n01534433',
                'cat': 'n02123045',
                'deer': 'n02415577',
                'dog': 'n02107683',
                'frog': 'n01644373',
                'horse': 'n02391049',
                'ship': 'n03095699',
                'truck': 'n03345487'}


class_dict = {v: k for k, v in class_dict.items()}

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Labels to condition the model
    with open('./misc/class_indices.txt', 'r') as fp:
        all_classes = fp.readlines()
    all_classes = [class_index.strip() for class_index in all_classes]
    if args.spec == 'woof':
        file_list = './misc/class_woof.txt'
    elif args.spec == 'nette':
        file_list = './misc/class_nette.txt'
    elif args.spec == 'cifar10':
        file_list = './misc/cifar10.txt'
    else:
        file_list = './misc/class100.txt'
    with open(file_list, 'r') as fp:
        sel_classes = fp.readlines()

    phase = max(0, args.phase)
    cls_from = args.nclass * phase
    cls_to = args.nclass * (phase + 1)
    sel_classes = sel_classes[cls_from:cls_to]
    sel_classes = [sel_class.strip() for sel_class in sel_classes]
    class_labels = []
    
    for sel_class in sel_classes:
        class_labels.append(all_classes.index(sel_class))


    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    

    main_ckpt_dir='logs/run-0/000-DiT-XL-2-minimax/checkpoints'
    ck_names = sorted(os.listdir(main_ckpt_dir), key=lambda x: int(x.split('.')[0]))
    ckpt_path = os.path.join(main_ckpt_dir, ck_names[-1])
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    batch_size = 1

    starting_time = time.time() 

    for class_label, sel_class in zip(class_labels, sel_classes):
        os.makedirs(os.path.join(args.save_dir, class_dict[sel_class]), exist_ok=True)
        for shift in tqdm(range(args.num_samples // batch_size)):
            # Create sampling noise:
            z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
            y = torch.tensor([class_label], device=device)

            # Setup classifier-free guidance:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * batch_size, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples / 0.18215).sample

            # Save and display images:
            for image_index, image in enumerate(samples):
                save_image(image, os.path.join(args.save_dir, class_dict[sel_class],
                                               f"{image_index + shift * batch_size + args.total_shift}.png"), normalize=True, value_range=(-1, 1))
                

    ending_time = time.time()
    running_time=ending_time - starting_time
    # save the running time
    with open(f'running_time_sample_ipc_{args.num_samples}.txt', 'w') as f:
        f.write(f'{running_time:.2f} seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spec", type=str, default='none', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='../logs/test', help='the directory to put the generated images')
    parser.add_argument("--num_samples", type=int, default=100, help='the desired IPC for generation')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=int, default=10, help='the class number for generation')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')
    args = parser.parse_args()
    main(args)
