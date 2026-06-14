import argparse
import json
import os

import mmengine
import numpy as np
from PIL import Image

import torch
import torch.distributed
import torch.utils.data
import tqdm
from transformers import AutoModel, AutoTokenizer

from projects.streamingrvos.evaluation.dataset import RefVOSDataset
from projects.streamingrvos.evaluation.utils import _init_dist_pytorch, _init_dist_slurm, get_dist_info, get_rank, collect_results_cpu

import concurrent.futures
from pycocotools import mask as cocomask


def async_func(executor, func, **kwargs):
    future = executor.submit(func, **kwargs)
    return future


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(cocomask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle


def mask_save(item, mask_prediction, work_dir):
    vid_id = item['video_id']
    exp_id = item['exp_id']
    save_path = os.path.join(work_dir, 'Annotations', vid_id, exp_id)
    mmengine.mkdir_or_exist(save_path)
    for id_m, mask in enumerate(mask_prediction):
        mask = Image.fromarray(mask.astype(np.float32) * 255).convert('L')
        file_name = item['frames'][id_m]
        save_file = os.path.join(save_path, file_name + ".png")
        mask.save(save_file)


DATASETS_INFO = {
    'DAVIS': {
        'data_root': '/mnt/data1/zwk/data/rvos/Ref-DAVIS17/',
        'image_folder': '/mnt/data1/zwk/data/rvos/Ref-DAVIS17/valid/JPEGImages/',
        'expression_file': '/mnt/data1/zwk/data/rvos/Ref-DAVIS17/meta_expressions/valid/meta_expressions.json',
        'mask_file': '/mnt/data1/zwk/data/rvos/Ref-DAVIS17/valid/mask_dict.pkl',
    },
    'MEVIS_U': {
        'data_root': '/mnt/data1/zwk/data/rvos/MeViS/valid_u',
        'image_folder': '/mnt/data1/zwk/data/rvos/MeViS/valid_u/JPEGImages',
        'expression_file': '/mnt/data1/zwk/data/rvos/MeViS/valid_u/meta_expressions.json',
        'mask_file': '/mnt/data1/zwk/data/rvos/MeViS/valid_u/mask_dict.json',
    },
    'REFYTVOS': {
        'data_root': '/mnt/data1/zwk/data/rvos/Ref-YTB-VOS/',
        'image_folder': '/mnt/data1/zwk/data/rvos/Ref-YTB-VOS/valid/JPEGImages/',
        'expression_file': '/mnt/data1/zwk/data/rvos/Ref-YTB-VOS/valid/meta_expressions_challenge.json',
        'mask_file': None,
    },
    'REREVOS': {
        'data_root': '/mnt/data1/zwk/data/rvos/ReVOS',
        'image_folder': '/mnt/data1/zwk/data/rvos/ReVOS/JPEGImages',
        'expression_file': '/mnt/data1/zwk/data/rvos/ReVOS/refering_split.json',
        'mask_file': None,
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='RefVOS')
    parser.add_argument('--model_path',type=str, default="Sa2VA-1B", help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_INFO.keys(),
        default='MEVIS',
        help='Specify a dataset')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--submit', action="store_true")
    parser.add_argument('--work_dir', type=str, default=None)
    parser.add_argument('--deepspeed', type=str, default=None) # dummy
    parser.add_argument('--record_frame_time', action='store_true')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


if __name__ == '__main__':
    args = parse_args()

    work_dir = args.work_dir
    if work_dir is None:
        work_dir = 'work_dirs/' +str(args.model_path) + "/" + str(args.dataset)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir, exist_ok=True)

    print("submit", args.submit)
    time_dir = None
    if args.record_frame_time:
        time_dir = os.path.join(work_dir, 'time_ms')
        os.makedirs(time_dir, exist_ok=True)

    if args.launcher == 'none':
        rank = 0
        world_size = 1
    elif args.launcher == 'pytorch':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
    elif args.launcher == 'slurm':
        _init_dist_slurm('nccl')
        rank, world_size = get_dist_info()

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset_info = DATASETS_INFO[args.dataset]


    dataset = RefVOSDataset(
        image_folder=dataset_info['image_folder'],
        expression_file=dataset_info['expression_file'],
        mask_file=dataset_info['mask_file'],
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,
        drop_last=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=2,
        pin_memory=False,
        collate_fn=lambda x:x[0],
    )
    results = []
    executor = concurrent.futures.ThreadPoolExecutor()
    for item in tqdm.tqdm(dataloader):
        frame_times_ms = None
        with torch.no_grad():
            # print(item['text_prompt'])
            # result = model.predict_forward_token(
            result = model.predict_forward(
                video=item['images'],
                text=item['text_prompt'],
                tokenizer=tokenizer,
            )
        if args.record_frame_time:
            frame_times_ms = result.get('frame_times_ms')
            if frame_times_ms is None:
                frame_times_ms = []
        # ytb my : len=19; 
        # ytb gf: result['prediction'][0].shape:[19, 720, 1280]
        text_idx = 0
        text_prediction = result['prediction']
        if len(result['prediction_masks']) > 0:
            mask_prediction = result['prediction_masks'][text_idx]
        else:
            print(text_prediction)
            mask_prediction = np.zeros((item['length'], item['ori_height'], item['ori_width']), dtype=np.uint8)

        if args.submit:
            async_func(executor, mask_save, item=item, mask_prediction=mask_prediction, work_dir=work_dir)
            encoded_mask = None
        else:
            encoded_mask = mask_to_rle(mask_prediction)

        result = {
            'index': item['index'],
            'video_id': item['video_id'],
            'exp_id': item['exp_id'],
            'text_prediction': text_prediction,
            'frames': item['frames'],
            'exp': item['text_prompt'],
            'prediction_masks': encoded_mask,

        }
        results.append(result)
        if args.record_frame_time:
            safe_vid = str(item['video_id']).replace('/', '_')
            safe_exp = str(item['exp_id']).replace('/', '_')
            time_path = os.path.join(time_dir, f'{safe_vid}_{safe_exp}.txt')
            with open(time_path, 'w', encoding='utf-8') as f:
                for frame_name, ms in zip(item['frames'], frame_times_ms):
                    f.write(f'{frame_name}\t{ms:.3f}\n')
                total_ms = sum(frame_times_ms)
                avg_ms = total_ms / max(len(frame_times_ms), 1)
                f.write(f'TOTAL_MS\t{total_ms:.3f}\n')
                f.write(f'AVG_MS\t{avg_ms:.3f}\n')


    executor.shutdown(wait=True)
    print(f'[Rank {rank}] : Finished.')
    
    if not args.submit:
        results = collect_results_cpu(results, len(dataset))
        if get_rank() == 0:
            final_results = {}
            for item in results:
                vid_id = item['video_id']
                exp_id = item['exp_id']
                if vid_id not in final_results:
                    final_results[vid_id] = {}
                assert exp_id not in final_results[vid_id]
                final_results[vid_id][exp_id] = item
            os.makedirs(work_dir, exist_ok=True)
            json.dump(final_results, open(f'{work_dir}/results.json', 'w'))

    if rank == 0:
        print('Done')
