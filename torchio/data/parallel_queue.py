import random
import warnings
from itertools import islice
from typing import List, Iterator

from tqdm import trange
from torch.utils.data import Dataset, IterableDataset, DataLoader

from .sampler import PatchSampler
from .dataset import SubjectsDataset
from functools import lru_cache as cache
from unsync import unsync, Unfuture
import time
import multiprocessing as mp
import torch
import numpy as np
from loguru import logger
import sys

class Counter(object):
    def __init__(self, initval=0):
        self.val = mp.Value('i', initval)
        self.lock = mp.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value

# these are tools for the command stream, they will not be exported
def shift_queue(queue, offset):
    return [(C,i+offset,L) for (C,i,L) in queue]

def randomize_queue(queue):
    num_subjects = sum(1 for C,idx,locks in queue if C=='L')
    new_indices = torch.randperm(num_subjects)
    randomized_queue = [(C,new_indices[i].item(), tuple(new_indices[l].item() for l in locks)) for C,i,locks in queue]
    return randomized_queue

def optimize_queue(queue):
    """ Very basic load queue optimizer.
        Whenever there is a delete operation,
        the next volume can be immediately loaded in the background.
        e.g. ... D P P L P D ... -> ... D L P P P D ...
    """

    queue = queue[:]
    insert = 0
    for idx in range(1,len(queue)-1):
        C,V,L = queue[idx]
        if C != 'D':
            continue
        for pos in range(idx+1, len(queue)):
            cmd = queue[pos]
            if cmd[0] == 'L':
                # there is a load after delete, we can move up the load part
                queue.pop(pos)
                queue.insert(idx+1,cmd)
                break
    return queue

def optimize_queue_prefetch(queue, prefetch=5):
    """ Very basic load queue optimizer.
        Whenever there is a delete operation,
        the next volume can be immediately loaded in the background.
        e.g. ... D P P L P D ... -> ... D L P P P D ...
    """
    while True:
        q_original = queue[:]
        for idx in range(1,len(queue)-1):
            C,V,L = queue[idx]
            C2,V2,L2 = queue[idx+1]

            if C=='P' and C2=='L': # loads can be propagated up
                queue[idx], queue[idx+1] = queue[idx+1], queue[idx]
                continue

            if C != 'D':
                continue

            if C=='D' and C2=='P' and V!=V2: # deletes can be propagated down
                queue[idx], queue[idx+1] = queue[idx+1], queue[idx]
                continue

            C,V,L = queue[idx]
            C2,V2,L2 = queue[idx+1] # idx+1 changed, reload is needed

            if len(L) >= prefetch or len(L2) >= prefetch:
                continue

            if C2=='L' and len(L2)<prefetch:# and len(L)<prefetch:
                queue[idx]   = (C2,V2,(V,)+L2)
                queue[idx+1] = (C, V, (V2,)+L)
        if q_original == queue:
            break

    return queue


def command_stream_generator(num_subjects, max_num_in_memory, num_patches, strategy='basic', randomize=True, optimize=True):
    if strategy == 'basic':
        queue = basic_command_stream_generator(num_subjects, max_num_in_memory, num_patches)
    elif strategy == 'merge_last_two':
        queue = merge_last_two(num_subjects, max_num_in_memory, num_patches)
    else:
        raise NotImplementedError

    if optimize:
        queue = optimize_queue(queue)

    if randomize:
        queue = randomize_queue(queue)
    return queue

def merge_last_two(num_subjects, max_num_in_memory, num_patches):
    N = num_subjects
    M = max_num_in_memory
    P = num_patches
    if N % M == 0 or N <= 2*M:
        return basic_command_stream_generator(N, M, P)
    R = N % M
    queue_head = basic_command_stream_generator(N - M - R, M, P)
    queue_tail = basic_command_stream_generator(M+R, M+R, P)
    return queue_head + shift_queue(queue_tail, N-M-R)

def basic_command_stream_generator(num_subjects, max_num_in_memory, num_patches):
    queue = []
    loaded_subjects = dict()
    all_subjects = set(range(num_subjects))
    deleted_subjects = set()
    while True:
        active_subjects = {x[0] for x in loaded_subjects.items() if x[1]>0}
        processed_subjects = set(loaded_subjects.keys())
        # load whenever you can
        if len(active_subjects) < max_num_in_memory and len(all_subjects) > len(processed_subjects):
            new_subject = (all_subjects - processed_subjects).pop()
            loaded_subjects[new_subject] = num_patches
            locks = tuple()
            queue.append(('L',new_subject, locks))
        else:
            # create patches when you cannot load new volume
            subject_list = list(active_subjects)
            new_order = torch.randperm(len(subject_list)).numpy().tolist()
            for idx in new_order:
                s = subject_list[idx]
                queue.append(('P',s, tuple()))
                loaded_subjects[s] -= 1
                # delete volume when it is exhausted
                if loaded_subjects[s] == 0:
                    queue.append(('D',s, tuple()))
                    deleted_subjects.add(s)
        # we are done where all volumes are deleted
        if len(all_subjects) == len(deleted_subjects):
            break
    return queue






def ParallelQueue(subjects_dataset: SubjectsDataset,
                  sampler: PatchSampler,
                  max_no_subjects_in_mem: int,
                  num_patches: int,
                  patch_queue_size=0,
                  seed=0,
                  double_buffer=True,
                ):
    if patch_queue_size == 0:
        patch_queue_size = max_no_subjects_in_mem*num_patches
    cmd_stream = command_stream_generator(len(subjects_dataset),
                                           max_no_subjects_in_mem,
                                            num_patches,
                                            strategy='basic',
                                            randomize=True,
                                            optimize=True)
    if double_buffer:
        cmd_stream = optimize_queue_prefetch(cmd_stream)
    return CommandStreamProcessingQueue(subjects_dataset, sampler, patch_queue_size, cmd_stream)


class CommandStreamProcessingQueue(IterableDataset):
    r"""Command stream processor and queue for patch-based training.
    """
    def __init__(
            self,
            subjects_dataset: SubjectsDataset,
            sampler: PatchSampler,
            patch_queue_size: int,
            command_stream: tuple,
            seed = 0,
            **kwargs
            ):
        self.subjects_dataset = subjects_dataset
        self.sampler = sampler

        if seed == 0:
            # we will need to set reproducible but different seeds for each subprocess
            self.seed = torch.randint(0,2**31, (1,)).item()
        else:
            self.seed = seed

        self.patch_queue = mp.Queue(patch_queue_size)
        self.initialize_with_command_stream(command_stream)

    @logger.catch
    def initialize_with_command_stream(self, command_stream):
        self.command_stream = command_stream
        self._pre_process_stream()

    @logger.catch
    def _pre_process_stream(self):
        self.samples_per_volume = {}
        self.length = 0
        for C,V,L in self.command_stream:
            if C=='P':
                self.length += 1
                self.samples_per_volume[V] = 1 + self.samples_per_volume.get(V,0)

    def __len__(self):
        return self.length

    @unsync(cpu_bount=True)
    @logger.catch
    def load(self, volume_idx, patch_per_volume, seed, locks):

        torch.manual_seed(seed)
        np.random.seed(seed)

        for idx in locks:
            while self.patch_counter_dict[idx].value()<1:
                time.sleep(0.1)

        logger.warning(f"volume is loading in the background: {volume_idx}")
        queue = self.queue_dict[volume_idx]
        volume = self.subjects_dataset[volume_idx]
        cnt = 0
        logger.warning(f"loading is finished: {volume_idx}")
        while cnt < patch_per_volume:
            for idx, sample in enumerate(self.sampler(volume)):
                if cnt >= patch_per_volume:
                    break
                logger.warning(f"volume  {volume_idx} patch {idx} is extracted")
                queue.put(sample)
                self.patch_counter_dict[volume_idx].increment()
                cnt += 1

    @logger.catch
    def get_patch(self, volume):
        queue = self.queue_dict[volume]
        return queue.get()

    @unsync(cpu_bount=True)
    @logger.catch
    def fill_queues(self, seed):
        i = 0
        pending_jobs = []
        volume_queues = {}
        logger.warning(f"inside the loop")
        for C,V,locks in self.command_stream:
            logger.warning(f"command {C} {V}")
            if C=='L': # load new volume, start sampler
                N = self.samples_per_volume[V]
                J = self.load(V, N, seed, locks)
                seed += 1
                pending_jobs.append(J)
                logger.warning(f"loaded")
                volume_queues[V] = J
            elif C=='D': # close volume
                pass
                # well, actually, the garbage collector will take care of this.
                #J = volume_queues[V]
                #J.result() # wait until the process exits

            elif C=='P': # extract patch
                #Take a patch from the volume's queue and put it into the global queue
                x = self.get_patch(V)
                logger.warning(f"try to insert value from {V}")
                self.patch_queue.put(x)
                logger.warning(f"{V} is inserted into the main queue")
            else:
                raise NotImplementedError
            logger.warning(f"command {C} {V} finished")

        logger.warning(f"trigger stream end in the main queue")
        self.patch_queue.put(None) # send message to the main thread that it can stop

        for job in pending_jobs:
            job.result()

    @logger.catch
    def __iter__(self):
        logger.warning(f"__iter__ called")
        self.queue_dict = {i:mp.Queue(8) for i in range(len(self.subjects_dataset))}
        self.patch_counter_dict = {i:Counter() for i in range(len(self.subjects_dataset))}

        logger.warning(f"queues set up")
        job = self.fill_queues(self.seed)
        logger.warning(f"iteration starts")
        logger.warning(job.done())
        while True:
            item = self.patch_queue.get()
            if item is not None:
                logger.error('item yielded')
                yield item
            else:
                break
        self.seed += 99991
        job.result()
