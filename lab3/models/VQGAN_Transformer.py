import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer

seed = 42
# Set the Python built-in random module seed
# random.seed(seed)
# Set the NumPy random seed
np.random.seed(seed)
# Set the PyTorch random seed for CPU
torch.manual_seed(seed)


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])
        self.prob_type = "ber"
    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    

##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, q_loss = self.vqgan.encode(x)
        return codebook_mapping, codebook_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        def linear(ratio):
            return 1.0 - ratio
        def cos(ratio):
            return np.cos(ratio * np.pi / 2)
        def square(ratio):
            return 1 -  np.square(ratio)
        if mode == "linear":
            return linear
        elif mode == "cosine":
            return cos
        elif mode == "square":
            return square
        else:
            raise NotImplementedError

##TODO2 step1-3:     
    def forward(self, x):
        _, z_indices=self.encode_to_z(x) #ground truth
        z_indices = z_indices.view(-1, self.num_image_tokens) # shape: (B, T)
        B,T = z_indices.shape
        # Step 2: Sample a random masking ratio via gamma scheduling
        mask_ratio = np.random.uniform(0, 1)
        if self.prob_type == "topk":
            # Top-k masking — randomly select num_keep tokens to keep
            num_keep = math.floor(mask_ratio * T)
            rand_scores = torch.rand(B, T, device=z_indices.device)
            topk_indices = rand_scores.topk(num_keep, dim=1).indices  # [B, num_keep]
            mask = torch.zeros(B, T, dtype=torch.bool, device=z_indices.device)
            mask.scatter_(1, topk_indices, True)  # True = keep, False = mask
        else:
            # Bernoulli sampling: mask each token independently
            mask = torch.bernoulli(torch.full_like(z_indices, 1 - mask_ratio, dtype=torch.float)).bool()  # True = keep
        # Step 5: Transformer prediction
        masked_input = z_indices.clone()
        masked_input[~mask] = self.mask_token_id  # only replace where False
        logits = self.transformer(masked_input) # shape [B, T, V]
        logits = logits[..., :self.mask_token_id]  # exclude mask token class from prediction
        #  One-hot encode ground truth for loss computation
        ground_truth = torch.zeros(z_indices.shape[0], z_indices.shape[1], self.mask_token_id).to(z_indices.device).scatter_(2, z_indices.unsqueeze(-1), 1)
        return logits, ground_truth
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self,image,ratio,mask_b):
         # Encode input to latent indices
        _, z_indices = self.encode_to_z(image)
        # Obtain logits from the transformer. Here the input is None as a placeholder.
        # Create a tensor filled with the mask token (broadcasted to the shape of z_indices)
        mask_token_tensor = torch.full_like(z_indices, self.mask_token_id)
        # Replace indices with the mask token wherever mask_b is active (== 1)
        z_indices_input = torch.where(mask_b == 1, mask_token_tensor, z_indices)
        logits = self.transformer(z_indices_input)
        # Apply softmax to convert logits into a probability distribution over the token vocabulary.
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        # Find, for each token position, the maximum probability and its corresponding token index.
        z_indices_predict_prob, z_indices_predict = torch.max(probabilities, dim=-1)
        # Define the ratio for masking. If the class has a 'ratio' attribute, use it; otherwise, default to 0.5.
        ratio = self.gamma(ratio)
        # Generate Gumbel noise to add temperature annealing into the confidence calculation.
        # The noise shape matches that of the predicted probabilities.
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob)))
        # Compute temperature: it scales the noise by (1 - ratio) and a preset choice_temperature.
        temperature = self.choice_temperature * (1 - ratio)
        # Combine the maximum probability with the temperature-scaled Gumbel noise to form a confidence measure.
        confidence = z_indices_predict_prob + temperature * g
        # For tokens that are not supposed to be inpainted (mask is False), set their confidence to infinity
        # so they remain unchanged by the prediction.
        confidence = torch.where(mask_b == 0,
                                torch.tensor(float('inf'), device=mask_b.device),
                                confidence)
        
        # Calculate the number of tokens to update based on the total number of masked tokens and the ratio.
        n = math.ceil(mask_b.sum().item()* ratio)
        
        # Flatten the confidence tensor so that we can use topk to select the tokens with lowest confidence.
        flat_confidence = confidence.view(-1)
        _, idx_to_mask = torch.topk(flat_confidence, n, largest=False)
        
        # Create a new mask (mask_bc) for the tokens to update.
        flat_mask_bc = torch.zeros_like(flat_confidence, dtype=mask_b.dtype)
        flat_mask_bc[idx_to_mask] = 1
        mask_bc = flat_mask_bc.view_as(mask_b)
        # Ensure that only originally masked tokens are updated by combining with the original mask.
        mask_bc = mask_bc * mask_b
        
        # Return the predicted token indices and the updated mask.
        return z_indices_predict, mask_bc

__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
