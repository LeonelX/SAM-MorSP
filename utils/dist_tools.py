import socket
import torch
import torch.distributed as dist

def gather_tensor(tensor, world_size):
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)

def gather_scalar(value, world_size, device):
    value_tensor = torch.tensor([value], device=device)
    value_list = [torch.zeros_like(value_tensor) for _ in range(world_size)]
    dist.all_gather(value_list, value_tensor)
    return torch.mean(torch.stack(value_list)).item()

def gather_dict_metrics(metric_dict, world_size, device):
    gathered = {}
    for k, v in metric_dict.items():
        if isinstance(v, (int, float)):
            gathered[k] = gather_scalar(v, world_size, device)
        else:
            raise TypeError(f"Metric '{k}' must be int or float, got {type(v)}")
    return gathered

def find_free_network_port() -> int:
    """
    Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # port
