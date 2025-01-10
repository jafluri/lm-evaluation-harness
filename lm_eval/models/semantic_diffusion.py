import os
import random
import logging
from typing import Literal

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from typing import Union
from lm_eval.utils import simple_parse_args_string

import torch
import torch.distributed as dist

from semantic_diffusion.utils import parse_dtype
from semantic_diffusion.loss import get_loss
from semantic_diffusion.checkpoints import load_checkpoint
from semantic_diffusion.likelihood import ELBO, compute_elbo


logger = logging.getLogger(__name__)

@register_model("semantic_diffusion")
class SemanticDiffusion(LM):
    def __init__(
        self,
        model_path: str,
        num_samples: int = 32,
        likelihood_method: Literal["loss", "elbo"] = "elbo",
        device: Union[str, torch.device] | None = None,
        batch_size: str | int = 1,
        **kwargs,
    ) -> None:
        # super init
        super().__init__()

        # attributes
        self.model_path = model_path
        self.num_samples = num_samples
        self.batch_size = int(batch_size)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "LOCAL_RANK" in os.environ:
            self._rank = int(os.environ["LOCAL_RANK"])
            self._world_size = int(os.environ["WORLD_SIZE"])
            dist.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
            torch.cuda.set_device(self.rank)
            self.device = torch.device("cuda", self.rank)

        # print the model path
        logger.info(f"[RANK: {self.rank}] Loading model from {model_path}")

        # # torch stuff
        # torch.set_float32_matmul_precision('high')

        # load the model
        model, noise_schedule, tokenizer, config = load_checkpoint(model_path, device=self.device)
        model.eval()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = parse_dtype(self.config.training.dtype)

        # parse the dtype the model was trained in
        self.dtype = parse_dtype(self.config.training.dtype)

        # construct the likelihood estimator
        loss_fn = get_loss(config, tokenizer, noise_schedule)
        if likelihood_method == "elbo":
            loss_fn.loss_weighting = True
            loss_fn.our_weighting = False
            loss_fn.min_loss_weight = -1e6
            loss_fn.max_loss_weight = 1e6
        likelihood = ELBO(config, model, noise_schedule, loss_fn)
        likelihood = likelihood.to(self.device)

        # compile for better efficiency
        self.likelihood = torch.compile(likelihood)

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        logger.info(additional_config)
        args = {}
        if additional_config is not None:
            args = additional_config
        args.update(simple_parse_args_string(arg_string))
        return cls(**args)

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        res = []
        
        # create the batches input
        strings = [r.args[0] + r.args[1] for r in requests]
        batched = [strings[i:i + self.batch_size] for i in range(0, len(strings), self.batch_size)] 
        for batch in tqdm(batched, disable=disable_tqdm):
            # load a batch
            bs = len(batch)
            if bs < self.batch_size:
                # pad to make sure we keep the same shape
                batch += [""] * (self.batch_size - bs)
            batch = self.tokenizer(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=self.config.model.max_seq_len)
            batch = batch.to(self.device)

            with torch.no_grad(), torch.autocast(self.device.type, self.dtype):            
                _, token_nlls = compute_elbo(
                    self.likelihood,
                    batch,
                    num_samples=self.num_samples,  # number of inner samples in the estimator, higher number has lower bias and variance
                    t_eps=self.config.model.get("t_eps", 1e-5),  # time epsilong for the noise schedule
                    show_progress=False,  # turn on/off the progress bar
                    return_token_nlls=True,  # set to True to return the token-level nlls
                )

                # shape of token_nlls: (batch_size, max_seq_len)
                # also includes NLL for padding tokens, can be masked like this:
                non_padding_nll = token_nlls * batch["attention_mask"]

                # get the NLL per sample
                nll = non_padding_nll.sum(dim=-1) / batch["attention_mask"].sum(dim=-1)
                # remove batch padding
                ll = -nll[:bs]
                is_greedy = True
                for x in ll.cpu().numpy():
                    res.append((x, is_greedy))
        
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

    @property
    def accelerator(self):
        return self._Accelerator(self.world_size)

    class _Accelerator:
        def __init__(self, world_size):
            self.world_size = world_size

        def wait_for_everyone(self):
            dist.barrier()

        def gather(self, local_tensor):
            gathered_tensors = [
                torch.zeros_like(local_tensor)
                for _ in range(self.world_size)
            ]
            dist.all_gather(gathered_tensors, local_tensor)
            if local_tensor.dim() < 1:
                return torch.stack(gathered_tensors)
            return torch.cat(gathered_tensors)
