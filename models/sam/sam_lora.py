import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import Sam
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from safetensors import safe_open
from safetensors.torch import save_file


class LoRA_qkv(nn.Module):
    """
    LoRA adaption for attention modules. Only for queries and values

    Arguments:
        qkv: Original block of attention
        linear_a_q: linear block for q
        linear_b_q: linear block for q
        linear_a_v: linear block for v
        linear_b_v: linear block for v

    Return:
        qkv(nn.Module): qkv block with all linear blocks added (equivalent to adding the matrix B*A)
    """

    def __init__(
            self,
            qkv,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        
        super(LoRA_qkv, self).__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.d_model = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x: Tensor):
        qkv = self.qkv(x)
        q_ba = self.linear_b_q(self.linear_a_q(x))
        v_ba = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :self.d_model] += q_ba #q part
        qkv[:, :, :, -self.d_model:] += v_ba #v part

        return qkv


class LoRA_sam(nn.Module):
    """
    Class that takes the image encoder of SAM and add the lora weights to the attentions blocks

    Arguments:
        sam_model: Sam class of the segment anything model
        rank: Rank of the matrix for LoRA
        lora_layer: List of weights exisitng for LoRA
    
    Return:
        None

    """

    def __init__(self, sam_model: Sam, rank: int, lora_layer=None):
        super(LoRA_sam, self).__init__()
        self.rank = rank
        assert rank > 0
        # 获取 sam_model 的设备和数据类型
        self.device = next(sam_model.parameters()).device
        self.dtype = next(sam_model.parameters()).dtype

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            # In each block, you have an attention block => total blocks -> nb lora layers
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))
        
        self.A_weights = []
        self.B_weights = []

        # freeze parameters of the image encoder
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # if only lora on few layers
            if t_layer_i not in self.lora_layer:
                continue

            w_qkv_linear = blk.attn.qkv
            self.d_model = w_qkv_linear.in_features

            w_a_linear_q = nn.Linear(self.d_model, self.rank, bias=False).to(self.device, self.dtype)
            w_b_linear_q = nn.Linear(self.rank, self.d_model, bias=False).to(self.device, self.dtype)
            w_a_linear_v = nn.Linear(self.d_model, self.rank, bias=False).to(self.device, self.dtype)
            w_b_linear_v = nn.Linear(self.rank, self.d_model, bias=False).to(self.device, self.dtype)
            

            self.A_weights.append(w_a_linear_q)
            self.B_weights.append(w_b_linear_q)
            self.A_weights.append(w_a_linear_v)
            self.B_weights.append(w_b_linear_v)

            blk.attn.qkv = LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v
            ).to(self.device, self.dtype)

        self.reset_parameters()
        self.sam = sam_model
        self.lora_vit = sam_model.image_encoder


    def reset_parameters(self):
        """
        Initialize the LoRA A and B matrices like in the paper
        """
        # Initalisation like in the paper
        for w_A in self.A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.B_weights:
            nn.init.zeros_(w_B.weight)
    
    def merge_lora(self):
        """
        Merge LoRA weights into the original QKV linear layers.
        """
        a_index = 0
        for t_layer_i, blk in enumerate(self.sam.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue

            # Get the original QKV linear layer
            original_qkv = blk.attn.qkv.qkv
            w_qkv_weight = original_qkv.weight.data

            # Get LoRA weights for Q and V
            w_a_linear_q = self.A_weights[a_index]
            w_b_linear_q = self.B_weights[a_index]
            w_a_linear_v = self.A_weights[a_index + 1]
            w_b_linear_v = self.B_weights[a_index + 1]

            # Calculate LoRA matrices for Q and V
            lora_q = torch.matmul(w_b_linear_q.weight.data, w_a_linear_q.weight.data)
            lora_v = torch.matmul(w_b_linear_v.weight.data, w_a_linear_v.weight.data)

            # Merge LoRA weights into QKV weights
            w_qkv_weight[:self.d_model] += lora_q
            w_qkv_weight[-self.d_model:] += lora_v

            # Replace the LoRA QKV module with the original QKV module
            blk.attn.qkv = original_qkv

            a_index += 2

        # Remove LoRA weights
        self.A_weights = []
        self.B_weights = []


    def save_lora_parameters(self, filename: str):
        """
        Save the LoRA wieghts applied to the attention model as safetensors.

        Arguments:
            filenmame: Name of the file that will be saved
        
        Return:
            None: Saves a safetensors file
        """
        num_layer = len(self.A_weights)
        # sufix 03:d -> allows to have a name 1 instead of 001
        a_tensors = {f"w_a_{i:03d}": self.A_weights[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.B_weights[i].weight for i in range(num_layer)}
        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)


    def load_lora_parameters(self, filename: str):
        """
        Load a safetensor file of LoRA weights for the attention modules

        Arguments:
            filename: Name of the file containing the saved weights
        
        Return:
            None: Loads the weights to the LoRA_sam class
        """
        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.A_weights):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.B_weights):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = nn.Parameter(saved_tensor)
    
    def save_lora_parameters(self, filename: str):
        """
        Save the LoRA weights applied to the attention model as a .pt file.

        Arguments:
            filenmame: Name of the file that will be saved

        Return:
            None: Saves a .pt file
        """
        num_layer = len(self.A_weights)
        # suffix 03:d -> allows to have a name 1 instead of 001
        a_tensors = {f"w_a_{i:03d}": self.A_weights[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.B_weights[i].weight for i in range(num_layer)}
        merged_dict = {**a_tensors, **b_tensors}
        torch.save(merged_dict, filename)
        
    
    def load_lora_parameters(self, filename: str):
        """
        Load a .pt file of LoRA weights for the attention modules

        Arguments:
            filename: Name of the file containing the saved weights

        Return:
            None: Loads the weights to the LoRA_sam class
        """
        state_dict = torch.load(filename)
        
        for i, w_A_linear in enumerate(self.A_weights):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = nn.Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.B_weights):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = nn.Parameter(saved_tensor)
            
    def save_parameters(self, save_path:str):
        """
        Save the LoRA and mask decoder weights applied to the attention model as a .pt file.

        Arguments:
            save_path: Name of the path that will be saved

        Return:
            None: Saves 2 .pt file
        """
        os.makedirs(save_path, exist_ok=True)
        # save mask decoder weights
        torch.save(self.sam.mask_decoder.state_dict(), os.path.join(save_path, 'm_dec_weights.pth'))
        # save lora weights
        self.save_lora_parameters(os.path.join(save_path, 'sam_lora_rank{:d}.pth'.format(self.rank)))
        
    def load_parameters(self, save_path: str):
        """
        Load a pt file of LoRA and mask decoder weights for the attention modules

        Arguments:
             save_path: Name of the path that weights pth

        Return:
            None
        """
        # load mask decoder weights
        self.sam.mask_decoder.load_state_dict(torch.load(os.path.join(save_path, 'm_dec_weights.pth')))
        # load lora weights
        self.load_lora_parameters(os.path.join(save_path, 'sam_lora_rank{:d}.pth'.format(self.rank)))



if __name__ == "__main__":
    sam = sam_model_registry["vit_b"](checkpoint="/home/xiej/data/models/sam_vit_b_01ec64.pth").cuda()
    params = sum(p.numel() for p in sam.parameters())
    print(f'The number of sam parameters is: {params}')
    lora_sam = LoRA_sam(sam,4)
    params = sum(p.numel() for p in lora_sam.parameters())
    print(f'The number of sam_lora parameters is: {params}')
    
    lora_sam.eval()
    input_image = torch.rand(size=(1,3,1024,1024)).cuda()
    with torch.no_grad():
        output0 = lora_sam.lora_vit(input_image)
    print(f'ViT encoder output shape {output0.shape}')
    
    # Merge LoRA weights
    lora_sam.merge_lora()
    params = sum(p.numel() for p in lora_sam.parameters())
    print(f'The number of sam_lora parameters after merging is: {params}')
    
    lora_sam.eval()
    with torch.no_grad():
        output1 = lora_sam.lora_vit(input_image)
    diff = nn.MSELoss()(output0, output1)
    print(f'L2 difference between original and merged weights: {diff.item()}')