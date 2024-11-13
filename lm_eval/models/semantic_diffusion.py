import random

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from typing import Union
from lm_eval.utils import simple_parse_args_string

import torch

from semantic_diffusion.forward_process import get_noise_schedule
from semantic_diffusion.utils import load_checkpoint, parse_dtype
from semantic_diffusion.likelihood import Likelihood, compute_nll


@register_model("semantic_diffusion")
class SemanticDiffusion(LM):
    def __init__(self, model_path: str, num_samples: int = 32, batch_size: int = 1) -> None:
        # super init
        super().__init__()

        # attributes
        self.model_path = model_path
        self.num_samples = num_samples
        self.batch_size = batch_size

        # print the model path
        print(f"Loading model from {model_path}")

        # torch stuff
        torch.set_float32_matmul_precision('high')
        torch.set_grad_enabled(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load the model
        self.model, self.tokenizer, self.config = load_checkpoint(model_path, device=self.device)
        self.model.eval()

        # parse the dtype the model was trained in
        self.dtype = parse_dtype(self.config.training.dtype)

        # load the noise schedule
        self.noise_schedule = get_noise_schedule(self.config, self.tokenizer)

        # construct the likelihood estimator
        likelihood = Likelihood(self.config, self.model, self.noise_schedule).to(self.device)

        # compile for better efficiency
        self.likelihood = torch.compile(likelihood)


    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        args = simple_parse_args_string(arg_string)
        return cls(**args)

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []
        
        # create the batches input
        strings = [r.args[0] + r.args[1] for r in requests]
        batched = [strings[i * self.batch_size:(i + 1) * self.batch_size] for i in range((len(strings) + self.batch_size - 1) // self.batch_size)] 
        with torch.no_grad(), torch.autocast(self.device.type, self.dtype):            
            for batch in tqdm(batched, disable=disable_tqdm):
                # load a batch
                batch = self.tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=self.config.model.max_seq_len)
                batch = batch.to(self.device)

                metrics, token_nlls = compute_nll(
                            self.likelihood,
                            batch,
                            num_samples=self.num_samples,  # number of inner samples in the estimator, higher number has lower bias and variance
                            show_progress=True,  # turn on/off the progress bar
                            return_token_nlls=True,  # set to True to return the token-level nlls
                        )

                # token-averaged NLL and PPL
                print(metrics["nll"], metrics["ppl"])

                # shape of token_nlls: (batch_size, max_seq_len)
                # also includes NLL for padding tokens, can be masked like this:
                non_padding_nll = token_nlls * batch["attention_mask"] 

                print(non_padding_nll)
                raise ValueError("gaga")
            # we set the greedy flag to False always
            res.append((-random.random(), False))
        
        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []

        for request in tqdm(requests, disable=disable_tqdm):
            res.append("lol")
            assert request.arguments[0].strip() != ""

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        res = []

        for _ in tqdm(requests, disable=disable_tqdm):
            res.append(-random.random())

        return res
