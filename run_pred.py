import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from dataloader import LongBenchManager, InfiniteBenchManager
from utils import DefaultDataCollator
from generation_utils import setup_caches

# from utils import DefaultDataCollator
from llama_utils import (
    llama_load_model_and_tokenizer,
    fastgen_generate,
)

import torch.multiprocessing as mp

from typing import List

import signal

timeout = 0


def handler(signum, frame):
    raise TimeoutError


signal.signal(signal.SIGALRM, handler)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def pred_loop_func(args, rank, task_queue, dataset_manager):
    seed_everything(args.seed)

    args.method = args.method.lower()
    model_load_kwargs = {}
    if args.method == "pyramidinfer":
        model_load_kwargs["pyramidinfer_config_file"] = args.pyramidinfer_config_file
    elif args.method == "keyformer":
        model_load_kwargs["model_maxlen"] = args.model_maxlen

    model_load_kwargs = {"device": f"cuda:{rank}"}
    model, tokenizer, generate_kwargs, _ = (
        llama_load_model_and_tokenizer(
            args.method, args.model_name_or_path, **model_load_kwargs
        )
    )
    cache_kwargs = {}
    cache_kwargs["max_cache_length"] = [1.0]
    cache_kwargs["global_tokens"] = 4
    cache_kwargs["hybrid_strategies"] = [
        {'strategy': 'special'},
        {'strategy': 'special_punc'},
        {'strategy': 'special_punc_heavy_hitter', 'heavy_hitter_frac': 0.3},
        {'strategy': 'special_punc_heavy_hitter_window', 'recent_window': 0.3, 'heavy_hitter_frac': 0.3},
        {'strategy': 'full'}
    ]
    cache_kwargs["cache_length_pattern"] = "tile"
    cache_kwargs["cache_strategy"] = ["hybrid"]
    cache_kwargs["prompt_compression_strategy"] = ["recent_global"]
    cache_kwargs["cache_strategy_pattern"] = "tile"
    cache_kwargs["recent_window"] = 10
    cache_kwargs["cache_bits"] = None
    cache_kwargs["feed_long_prompts"] = False
    cache_kwargs["history_window_size"] = 1
    cache_kwargs["attn_thresholding"] = False
    cache_kwargs["min_recovery_frac"] = 0.9
    setup_caches(model, generate_kwargs["special_ids"], generate_kwargs["punc_ids"], rank, 32000, cache_kwargs)

    print(
        f"Worker {rank} use GPU[{rank}] mem: {torch.cuda.max_memory_allocated(id) / 1024 ** 3:.2f} GB"
    )

    try:
        signal.alarm(timeout)  # start timer
        while True:
            data = task_queue.get()
            if data is None:
                break
            warm_up, dataset_name, x, generate_kwargs = data

            indices = x.pop("index").tolist()
            x["input_ids"] = x["input_ids"].to(rank)
            x["attention_mask"] = x["attention_mask"].to(rank)
            answers = x.pop("answers")
            length = x.pop("length")
            all_classes = x.pop("all_classes")

            outputs = fastgen_generate(x, generate_kwargs, model, tokenizer)

            if not warm_up:
                for pred, answer, c, l in zip(outputs, answers, all_classes, length):
                    dataset_manager.write_one_result_v2(
                        args.output_dir, pred, answer, c, l.item(), dataset_name
                    )

            # reset timer
            signal.alarm(timeout)

        signal.alarm(0)
    except TimeoutError:
        print("The operation timed out. Exiting!")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_maxlen", type=int, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="longbench",
        choices=["longbench", "infinitebench"],
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pyramidinfer_config_file", type=str, default="")
    parser.add_argument(
        "--method",
        type=str,
        default="full",
        choices=["full", "streamingllm", "keyformer", "h2o", "pyramidinfer", "kvclus", "kvclus-eager", "kvclus-fullscore", "kvclus-hash"],
    )
    parser.add_argument("--write_in_time", action="store_true")
    parser.add_argument("--mp_num", default=1, type=int)
    parser.add_argument("--pp_num", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--min_seq_len", type=int, default=16000)
    args = parser.parse_args()

    print(args)
    seed_everything(args.seed)

    if args.dataset_name == "longbench":
        dataset_manager = LongBenchManager(
            args.dataset_path,
            args.dataset_path,
            "test",
            args.e,
            args.retrieval,
        )
        tasks = dataset_manager.get_dataset_names(with_e=args.e, retrieval=args.retrieval)
    else:
        dataset_manager = InfiniteBenchManager(
            args.dataset_path,
            args.dataset_path,
        )
        tasks = dataset_manager.get_dataset_names()
    print("datasets: ", tasks)

    # load model and tokenizer
    args.method = args.method.lower()
    model_load_kwargs = {"device": "cpu"}
    model, tokenizer, generate_kwargs, apply_chat_template = (
        llama_load_model_and_tokenizer(
            args.method, args.model_name_or_path, **model_load_kwargs
        )
    )

    # set generate_kwargs
    if hasattr(model, "generation_config"):
        eos_token_id = model.generation_config.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    del model

    # start processes
    task_queue = mp.Queue(maxsize=args.mp_num)
    work_processes = []
    for i in range(args.mp_num):
        p = mp.Process(
            target=pred_loop_func,
            args=(args, i, task_queue, dataset_manager),
        )
        p.start()
        work_processes.append(p)

    warm_up = True

    for dataset_name in tasks:
        raw_data = dataset_manager.get_data(dataset_name)
        _, dataset_maxlen, dataset_category = dataset_manager.get_dataset_info(
            dataset_name
        )

        process_fn = partial(
            dataset_manager.process_raw_data,
            tokenizer=tokenizer,
            apply_chat_template=apply_chat_template,
            task=dataset_name,
            max_length=args.model_maxlen,
            truncate_from_middle=True,
        )

        remove_columns = []
        for key in raw_data[0]:
            if key not in ["length", "all_classes", "answers"]:
                remove_columns.append(key)
        encoded_data = raw_data.map(
            process_fn,
            batched=True,
            num_proc=4,
            batch_size=10,
            with_indices=True,
            remove_columns=remove_columns,
        )

        dataloader = torch.utils.data.DataLoader(
            encoded_data,
            batch_size=args.batch_size,
            collate_fn=DefaultDataCollator(tokenizer=tokenizer),
            pin_memory=False,
        )

        generate_kwargs["max_new_tokens"] = dataset_maxlen
        generate_kwargs["eos_token_id"] = eos_token_id
        if dataset_name in [
            "2wikimqa",
            "hotpotqa",
            "musique",
            "multifieldqa_en",
            "qasper",
            "narrativeqa",
            "samsum",
        ]:
            if dataset_category is not None and "QA" in dataset_category:
                generate_kwargs["eos_token_id"].append(
                    tokenizer.encode("\n", add_special_tokens=False)[-1]
                )

        if warm_up:
            for i, x in enumerate(tqdm(dataloader, desc=f"Warm up")):
                if x["input_ids"].size(-1) < args.min_seq_len:
                    continue
                for _ in range(args.mp_num):
                    task_queue.put((warm_up, dataset_name, x, generate_kwargs))
                warm_up = False
                break
        
        if warm_up:
            continue
        
        for i, x in enumerate(tqdm(dataloader, desc=f"Send tasks for {dataset_name}")):
            if x["input_ids"].size(-1) < args.min_seq_len:
                continue
            task_queue.put((warm_up, dataset_name, x, generate_kwargs))

    for _ in range(args.mp_num):
        task_queue.put(None)

    for p in work_processes:
        p.join()
