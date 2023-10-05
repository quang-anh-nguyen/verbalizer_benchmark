import logging
import random
import numpy as np
import torch

def get_logger(name, filename=None):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handlers = [logging.StreamHandler()]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(name)s in %(funcName)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    for handler in handlers:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    
    """
    Code adpated from https://github.com/yuezhihan/ts2vec/blob/main/utils.py
    """
    
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]


def postprocess_prediction(raw, metrics):
    if not isinstance(raw, dict):
        raw = raw._asdict()
    result = {
        'logits': None if len(raw['predictions'].shape)<=1 else raw['predictions'].tolist(),
        'predictions': raw['predictions'].tolist() if len(raw['predictions'].shape)<=1 else raw['predictions'].argmax(-1).tolist(),
        'loss': raw['metrics']['test_loss'],
        'metrics': {
            metric: raw['metrics']['test_'+metric] for metric in metrics
        },
        'time': {
            num: raw['metrics']['test_'+num] for num in ['runtime', 'samples_per_second', 'steps_per_second']
        }
    }
    return result

def add_to_list(value, dic, key):
    if key in dic:
        dic[key].append(value)
    else:
        dic[key] = [value]

def postprocess_history(log_history, metrics):
    result = {
        'train': {},
        'valid': {}
    }
    for e in log_history:
        if any([k.startswith('eval_') for k in e]):
            for k in metrics:
                add_to_list(e['eval_'+k], result['valid'], k)
            if 'eval_score' in e:
                add_to_list(e['eval_score'], result['valid'], 'score')
        else:
            if 'loss' in e:
                add_to_list(e['loss'], result['train'], 'loss')
            else:
                result['train']['time'] = {
                    'runtime': e['train_runtime'],
                    'samples_per_second': e['train_samples_per_second'],
                    'steps_per_second': e['train_steps_per_second']
                }
    return result

def remove_logger(name):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.disabled = True
