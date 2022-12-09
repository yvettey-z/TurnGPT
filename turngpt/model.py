# from __future__ import annotations

from argparse import ArgumentParser
from typing import Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
import numpy as np

from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from turngpt.generation import generate, generate_greedy_from_tokenized
from turngpt.plot_utils import plot_trp
from turngpt.projection_labeler import ProjectionLabeler
from turngpt.tokenizer_rev2 import tokenizer_AMI
from turngpt.tokenizer import SpokenDialogTokenizer
from turngpt.eval import calc_score, process_for_nltk


mpl.use("agg")


def load_transformer(
    pretrained_model_name_or_path="gpt2", pretrained=True, **model_kwargs
):
    """Load transformer model. If `pretrained` then initialize with pretrained weights otherwise start from scratch"""

    implemented = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
    ]

    update_on_pretrain = ["embd_pdrop", "attn_pdrop", "resid_pdrop"]

    if not (
        "gpt2" in pretrained_model_name_or_path.lower()
        or "dialogpt" in pretrained_model_name_or_path.lower()
    ):
        raise NotImplementedError(
            f"pretrained_model_name_or_path=`{pretrained_model_name_or_path}` is Not implemented! Please use: {implemented}"
        )

    config = GPT2Config.from_pretrained(pretrained_model_name_or_path)
    if pretrained:
        # Update only certain parameters
        for k, v in model_kwargs.items():
            if k in update_on_pretrain and v is not None:
                config.update({k: v})
        transformer = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, config=config
        )
    else:
        for k, v in model_kwargs.items():
            if v is not None:
                config.update({k: v})
        transformer = GPT2LMHeadModel(config=config)
    return transformer


class Utils:
    tokenizer: Union[SpokenDialogTokenizer, tokenizer_AMI]

    def idx_to_string(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        s = self.tokenizer.convert_ids_to_tokens(idx)
        s = self.tokenizer.convert_tokens_to_string(
            s.strip()
        )  # remove prefix space/symbol
        return s

    def get_trp(self, x):
        return x[..., self.tokenizer.eos_token_id]

    def tokenize_strings(self, string_or_list, add_post_eos_token=False):
        if isinstance(string_or_list, str) and add_post_eos_token:
            if not string_or_list.strip().endswith(self.tokenizer.eos_token):
                string_or_list += self.tokenizer.eos_token

        if isinstance(self.tokenizer, SpokenDialogTokenizer):
            t = self.tokenizer(string_or_list, return_tensors = "pt")
        else:
            t = self.tokenizer.tokenize_old(string_or_list, return_tensors="pt")

        if not isinstance(t["input_ids"], torch.Tensor):
            tmp_inp, tmp_sp = [], []
            for inp, sp in zip(t["input_ids"], t["speaker_ids"]):
                tmp_inp.append(torch.tensor(inp))
                tmp_sp.append(torch.tensor(sp))
            tmp = self.tokenizer.pad({"input_ids": tmp_inp})
            t["input_ids"] = tmp["input_ids"]
            t["attention_mask"] = tmp["attention_mask"]
            t["speaker_ids"] = self.tokenizer.pad({"input_ids": tmp_sp})["input_ids"]

        for k, v in t.items():
            t[k] = v.to(self.device)
        return t

    def get_tokens(self, input_ids):
        def inner(input_ids):
            inner_tokens = []
            for idx in input_ids:
                inner_tokens.append(self.idx_to_string(idx))
            return inner_tokens

        def outer(input_ids):
            tokens = []
            for batch in input_ids:
                tokens.append(inner(batch))
            return tokens

        tokens = None
        if isinstance(input_ids, torch.Tensor):
            if input_ids.ndim > 2:
                raise LookupError(
                    f"self.get_tokens not implemented for Tensor shape: {tuple(input_ids.shape)} (>2)"
                )
            elif input_ids.ndim == 2:
                tokens = outer(input_ids)
            else:
                tokens = inner(input_ids)
                for idx in input_ids:
                    tokens.append(self.idx_to_string(idx))
        elif isinstance(input_ids, list):
            if isinstance(input_ids[0], list):
                tokens = outer(input_ids)
            else:
                tokens = inner(input_ids)
        return tokens

    @torch.no_grad()
    def string_list_to_trp(
        self, string_or_list, add_post_eos_token=False, **model_kwargs
    ):
        t = self.tokenize_strings(string_or_list, add_post_eos_token=add_post_eos_token)

        # Model
        out = self(t["input_ids"], speaker_ids=t["speaker_ids"], **model_kwargs)
        out["probs"] = out["logits"].softmax(dim=-1)
        out["trp_probs"] = self.get_trp(out["probs"])
        out["tokens"] = self.get_tokens(t["input_ids"])
        if "mc_logits" in out:
            out["trp_proj"] = out["mc_logits"].sigmoid()
        return out


class TurnGPTWandbCallbacks(pl.Callback):
    turn_list = [
        ["yesterday we met in the park", "okay when will you meet again", "tomorrow"],
        [
            "Hello there I basically had the worst day of my life",
            "Oh no, what happened?",
            "Do you want the long or the short story?",
        ],
    ]

    def __init__(
        self,
        text_list=None,
        n_steps=200,
        n_generate=20,
        eos_token="<ts>",
        unk_token="<|endoftext|>",
    ):
        super().__init__()
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.text_list = text_list
        self.n_steps = n_steps
        self.n_generate = n_generate
        if self.text_list is None:
            self.text_list = self.turn_list

    def trp_plots(self, trainer, pl_module, name="TRP/example"):
        out = pl_module.string_list_to_trp(self.text_list)

        for b in range(out["trp_probs"].shape[0]):
            proj = out["trp_proj"][b].cpu() if "trp_proj" in out else None
            fig, _ = plot_trp(
                trp=out["trp_probs"][b].cpu(),
                proj=proj,
                text=out["tokens"][b],
                unk_token=pl_module.tokenizer.unk_token,
                eos_token=pl_module.tokenizer.eos_token,
                plot=False,
            )

            pl_module.logger.experiment.log(
                data={
                    f"{name}_{b}": wandb.Image(fig),
                    "global_step": trainer.global_step,
                },
            )
            plt.close("all")

    def generate(self, trainer, pl_module, name):
        gen = generate(
            pl_module,
            context=self.turn_list[-1],
            n_steps=self.n_steps,
            top_p=0.9,
            top_k=-1,
            n_trajectories=self.n_generate,
            strategy="sample",
            stop_at_eos=True,
        )
        # remove duplicates
        l = (gen["input_ids"][0] != -1).sum()
        G = {"tokens": [gen["tokens"][0]], "probs": [gen["probs"][0][:l].cpu()]}
        for i, g in enumerate(gen["tokens"][1:]):
            if g not in G["tokens"]:
                l = (gen["input_ids"][i] != -1).sum()
                G["tokens"].append(g)
                G["probs"].append(gen["probs"][i][:l].cpu())

        table = wandb.Table(
            columns=["context", "sample", "probs"],
            data=[
                ["... " + self.turn_list[-1][-1], toks, probs.tolist()]
                for toks, probs in zip(G["tokens"], G["probs"])
            ],
        )
        pl_module.logger.experiment.log(
            data={
                f"{name}": table,
                "global_step": trainer.global_step,
            },
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        self.trp_plots(trainer, pl_module, name="TRP/example")
        self.generate(trainer, pl_module, name="Gen")

    def on_save_checkpoint(self, trainer, pl_module, *args, **kwargs):
        self.trp_plots(trainer, pl_module, name="TRP-chpt/example")
        self.generate(trainer, pl_module, name="Gen-chpt")


class TurnGPT(pl.LightningModule, Utils):
    """
    This is the code example of teaching and research.

    Add features to this model i.e. analysis of turn-taking.


    On training
    * call `model.initialize_special_embeddings()` to initialize <ts> = eos_token
    """

    def __init__(
        self,
        pretrained_model_name_or_path="gpt2",
        pretrained=True,
        trp_projection_steps=-1,
        trp_projection_type="linear",
        omit_dialog_states=False,
        no_train_first_n=5,
        learning_rate=1e-4,
        weight_loss=False,
        weight_regular_token=0.5,
        weight_eos_token=1.0,
        weight_decay=0.0,
        dropout=None,
        num_speakers=2,
        **model_kwargs,
    ):
        super().__init__()
        self.name_or_path = pretrained_model_name_or_path
        self.pretrained = pretrained

        # train parameters
        self.no_train_first_n = no_train_first_n
        self.learning_rate = learning_rate
        self.weight_loss = weight_loss
        self.weight_regular_token = weight_regular_token
        self.weight_eos_token = weight_eos_token
        self.omit_dialog_states = omit_dialog_states
        self.weight_decay = weight_decay
        self.num_speakers = num_speakers

        # Configure dropout rates
        if dropout is not None:
            model_kwargs["resid_pdrop"] = dropout
            model_kwargs["embd_pdrop"] = dropout
            model_kwargs["attn_pdrop"] = dropout

        # Load `transformers` model
        self.transformer = load_transformer(
            pretrained_model_name_or_path, pretrained=pretrained, **model_kwargs
        )

        # TRP projection head
        self.trp_projection_steps = trp_projection_steps
        self.train_accuracy, self.valid_accuracy, self.test_accuracy = None, None, None
        if trp_projection_steps > 0:
            self.trp_projection_type = trp_projection_type
            hidden_size = self.transformer.config.hidden_size
            task = 'multiclass' if self.num_speakers > 2 else 'binary'
            self.train_accuracy = torchmetrics.Recall(task=task, average='macro',        # might change the var names
                                                        num_classes=self.num_speakers, top_k=1)
            self.valid_accuracy = torchmetrics.Recall(task=task, average='macro',
                                                        num_classes=self.num_speakers, top_k=1)
            self.test_accuracy = torchmetrics.Recall(task=task, average='macro',
                                                       num_classes=self.num_speakers, top_k=1)

            # MultiTask Head operating on n last hidden states
            if trp_projection_type.lower() == "attention":
                raise NotImplementedError()
            else:
                self.trp_projection_head = nn.Sequential(
                    nn.Dropout(p=dropout) if dropout is not None else nn.Identity(),
                    )
                if self.num_speakers > 2:
                    self.trp_projection_head = nn.Sequential(
                        nn.Dropout(p=dropout) if dropout is not None else nn.Identity(),
                        nn.Linear(hidden_size, self.num_speakers)
                    )
                else:
                    self.trp_projection_head = nn.Sequential(
                        nn.Dropout(p=dropout) if dropout is not None else nn.Identity(),
                        nn.Linear(hidden_size, 1))
                # nn.Sequential doesn't have attribute as "append"

        self.tokenizer = None # None until calling init_tokenizer

        self.gen_scores = {'BLEU-2': 0, 'BLEU-4': 0, 'METEOR': 0, 'NIST-2': 0, 'NIST-4': 0, 'count': 0} # Only for test

        self.save_hyperparameters()

    @property
    def run_name(self):
        name = "TurnGPT"
        if self.trp_projection_steps > 0:
            name += f"_proj_{self.trp_projection_steps}"
        return name

    def init_tokenizer(self):
        # The tokenizer should always be a part of the model
        if self.num_speakers == 2:
            self.tokenizer = SpokenDialogTokenizer(self.name_or_path)
        else:
            self.tokenizer = tokenizer_AMI()

        # Add extra embeddings for custom tokens
        # Optional: Initialize <ts> to be close to punctuation tokens.
        self.transformer.resize_token_embeddings(new_num_tokens=len(self.tokenizer))

    def initialize_special_embeddings(self, tokens=["!", "?", "."]):
        """
        Initialize `eos_token` as the average of `tokens`.

        By default (or looking at <speaker1/2>) the embeddings are initalized to m=0, std=0.02
        """
        ts = self.tokenizer.eos_token_id
        # pre = self.transformer.transformer.wte.weight[ts].clone()
        with torch.no_grad():
            ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
            avg_emb = self.transformer.transformer.wte(ids).mean(0)
            self.transformer.transformer.wte.weight.data[ts] = avg_emb
        # post = self.transformer.transformer.wte.weight[ts]
        # print(pre == post)
        print(f"Initalized {self.tokenizer.eos_token} -> avg({tokens})")

    def print_parameters(self):
        print("")
        print("TurnGPT")
        print("name_or_path: ", self.name_or_path)
        print("learning_rate: ", self.learning_rate)
        print("weight_loss: ", self.weight_loss)
        if self.weight_loss:
            print("weight_regular_token: ", self.weight_regular_token)
            print("weight_eos_token: ", self.weight_eos_token)
        if self.trp_projection_steps > 0:
            print("trp_projection_steps: ", self.trp_projection_steps)
            print("trp_projection_type: ", self.trp_projection_type)
        print()

    def get_labels(self, input_ids, mask, value=-100):
        """Don't shift the labels (happens internally)"""
        labels = input_ids.clone()
        labels[torch.logical_not(mask)] = value

        if self.no_train_first_n > 0:
            labels[:, : self.no_train_first_n] = value
        return labels

    def get_projection_labels(self, input_ids, mask, value=-100):
        """match with speaker id, instead of eos token id"""
        labeler = ProjectionLabeler(
            projection_steps=self.trp_projection_steps,
            token_id=self.tokenizer.eos_token_id,
        ).to(self.device)
        proj_labels = labeler(input_ids)
        proj_labels[torch.logical_not(mask)] = value
        if self.no_train_first_n > 0:
            proj_labels[:, : self.no_train_first_n] = value
        return proj_labels

    @torch.no_grad()
    def get_loss_weight(self):
        weight = (
            torch.ones(len(self.tokenizer), dtype=torch.float)
            * self.weight_regular_token
        )
        weight[self.tokenizer.eos_token_id] = self.weight_eos_token
        return weight.to(self.device)

    def cross_entropy_loss(self, logits, labels, reduction="mean"):
        """
        Taken from GPT2LMHeadModel in:

          https://github.com/huggingface/transformers/blob/91ff480e2693f36b11aaebc4e9cc79e4e3c049da/src/transformers/models/gpt2/modeling_gpt2.py#L968

        How to weight CE-Loss?

            https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10

        Using the custom weights gets a total loss that is larger than without?
        I don't want the model to train as much on the new type of text it receives but to learn the `eos_token` better.
        I assume that the loss should be less if I scale down the normal tokens loss...

        `CrossEntropyLoss` seems to normalize the loss if not `reduction=None` which account for the above behaviour.

        Instead we use `reduction=none` and simply average over the weighted loss values.
        Given that `weight_regular_token` < 1 and `weight_eos_token` >=1 we get a lower loss when weighting.

        Is this mathematically sound? well that is the question.

        I guess one could simply scale the `weight_eos_token > 1` while `weight_regular_token=1` and use a smaller learning rate?
        """
        weight = None
        if self.weight_loss:
            weight = self.get_loss_weight()

        loss_fct = nn.CrossEntropyLoss(weight=weight, reduction="none")

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        indices_for_training = shift_labels != -100

        # Flatten the tokens and calc loss
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        if reduction != "none":
            loss = torch.sum(loss * torch.reshape(indices_for_training, (-1,))) / torch.sum(indices_for_training)
            loss = loss.mean()
        return loss

    def ce_loss(self, logits, labels):
        """
        Extended simple BCELoss from binary trp projection to multi-class trp projection

        Input: num_speakers, 2 by default
        """
        if self.num_speakers == 2:
            loss_fct = nn.BCEWithLogitsLoss()

            shift_logits = logits[..., :-1]   #.contiguous()
            shift_labels = labels[..., 1:]    #.contiguous()

        else:
            loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()   # ?

        # print("shift_labels:", shift_labels.shape) # (16, 499)
        # print("shift_logits:", shift_logits.shape) # (16, 500, 3)
        # Manually select appropriate steps
        # Omit steps where label is -100 (like CrossEntropyLoss)
        indices_for_training = shift_labels != -100

        if self.num_speakers == 2:
            loss = loss_fct(
                torch.masked_select(shift_logits, indices_for_training),
                torch.masked_select(shift_labels, indices_for_training),
            )
        else:
            #indices_for_training_expanded = indices_for_training.unsqueeze(-1).expand(shift_logits.shape).type(torch.uint8)
            #print("indices_for_training_expanded:", indices_for_training_expanded.shape)
            #mask_logits = torch.masked_select(shift_logits, indices_for_training_expanded)
            #mask_labels = torch.masked_select(shift_labels, indices_for_training)

            # print("loss_fct[0]:", mask_logits.shape)
            # print("loss_fct[0]:", mask_labels.shape)
            loss = loss_fct(
                # torch.masked_select(shift_logits, indices_for_training_expanded).reshape((-1, self.num_speakers)),
                torch.reshape(shift_logits, (-1, self.num_speakers)),
                torch.reshape(shift_labels, (-1,))
            )

            loss = torch.sum(loss * torch.reshape(indices_for_training, (-1,))) / torch.sum(indices_for_training)
        # shift_logits = torch.masked_select(shift_logits, indices_for_training)
        # shift_labels = torch.masked_select(shift_labels, indices_for_training)
        # loss = loss_fct(shift_logits, shift_labels)
        return loss

    def get_likelihood(self, logits, labels, pad_last=True, pad_first=False):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_probs = shift_logits.softmax(dim=-1)

        # likelihood = shift_probs[shift_labels]
        bn = torch.ones_like(shift_labels)
        bn[0] = 0

        seq = torch.arange(shift_labels.shape[-1])
        seq = torch.stack([seq] * shift_labels.shape[0])

        likelihood = shift_probs[bn.view(-1), seq.view(-1), shift_labels.view(-1)]
        likelihood = likelihood.view(shift_labels.shape)
        if pad_first:
            likelihood = torch.cat(
                [torch.zeros(likelihood.shape[0], 1), likelihood], dim=-1
            )
        elif pad_last:
            likelihood = torch.cat(
                [likelihood, torch.zeros(likelihood.shape[0], 1)], dim=-1
            )
        return likelihood

    def forward(
        self,
        input_ids=None,
        speaker_ids=None,
        num_speakers=None,
        labels=None,
        mc_labels=None,
        use_cache=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """
        Simple rewrite of original:

            https://github.com/huggingface/transformers/blob/439a43b6b403205eeda2d62645fc16c93627d30d/src/transformers/models/gpt2/modeling_gpt2.py#L1086
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.transformer.config.use_return_dict
        )

        transformer_outputs = self.transformer.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=speaker_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        num_speakers = self.num_speakers

        # Set device for model parallelism
        if self.transformer.model_parallel:
            torch.cuda.set_device(self.transformer.transformer.first_device)
            hidden_states = hidden_states.to(self.transformer.lm_head.weight.device)

        # Language Modeling
        lm_logits = self.transformer.lm_head(hidden_states)
        lm_loss = None
        if labels is not None:
            lm_loss = self.cross_entropy_loss(lm_logits, labels)

        # MultiTask Modeling
        mc_logits = None
        mc_loss = None
        if self.trp_projection_steps > 0:
            # NOTE:
            # Assumed to only guess a single class
            if num_speakers == 2:
                mc_logits = self.trp_projection_head(hidden_states).squeeze(-1)   # still need to check
            else:
                mc_logits = self.trp_projection_head(hidden_states)

            if mc_labels is not None:
                if num_speakers == 2:
                    mc_loss = self.ce_loss(mc_logits, mc_labels)
                else:
                    mc_loss = self.ce_loss(mc_logits, speaker_ids)

        # if not return_dict:
        #     output = (lm_logits, mc_logits) + transformer_outputs[1:]
        #     if mc_loss is not None:
        #         output = (mc_loss,) + output
        #     return ((lm_loss,) + output) if lm_loss is not None else output

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def configure_optimizers(self):
        # NOTE:
        # Use multiple optimizers for transformer and projection?
        # see:
        #   https://pytorch-lightning.readthedocs.io/en/stable/common/optimizers.html#use-multiple-optimizers-like-gans-manual
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def on_save_checkpoint(self, checkpoint):
        """We must save the tokenizer used during training"""
        checkpoint["tokenizer"] = self.tokenizer

    def on_load_checkpoint(self, checkpoint):
        """We must load the tokenizer used during training and resize the embeddings appropriately"""
        if "tokenizer" in checkpoint:
            print("#" * 70)
            print("LOAD CHECKPOINT TOKENIZER")
            self.tokenizer = checkpoint["tokenizer"]
            print("Loaded tokenizer")
            print(self.tokenizer)

            # Add extra embeddings for custom tokens
            self.transformer.resize_token_embeddings(new_num_tokens=len(self.tokenizer))
            print("Resized weights")
            print("#" * 70)

    def training_step(self, batch, batch_idx):
        lm_labels = self.get_labels(batch["input_ids"], mask=batch["attention_mask"])

        proj_labels = None
        if self.trp_projection_steps > 0:
            proj_labels = self.get_projection_labels(
               batch["input_ids"], mask=batch["attention_mask"]
            )

        if self.omit_dialog_states:
            batch["speaker_ids"] = None

        out = self.forward(
            batch["input_ids"],
            speaker_ids=batch["speaker_ids"],
            labels=lm_labels,
            mc_labels=proj_labels,
            attention_mask=batch["attention_mask"],
        )

        if self.trp_projection_steps > 0:
            self.log("loss_lm", out["loss"])
            self.log("loss_projection", out["mc_loss"])
            total_loss = out["loss"] + out["mc_loss"]
            return {"loss": total_loss, "mc_logits": out["mc_logits"].detach(), "mc_labels": proj_labels.detach()}
        else:
            self.log("loss", out["loss"])
            total_loss = out["loss"]
            return {"loss": total_loss}

    def training_step_end(self, step_output):
        if self.trp_projection_steps > 0:
            shift_logits, shift_labels = self.shift_logits_labels(step_output["mc_logits"], step_output["mc_labels"])
            self.train_accuracy(shift_logits, shift_labels)
            self.log("bAcc", self.train_accuracy, prog_bar=True, logger=False)
    
    def training_epoch_end(self, outputs):
        if self.trp_projection_steps > 0:
            self.log("train_bAcc_epoch", self.train_accuracy.compute())
            self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        lm_labels = self.get_labels(batch["input_ids"], mask=batch["attention_mask"])

        proj_labels = None
        if self.trp_projection_steps > 0:
            proj_labels = self.get_projection_labels(
                batch["input_ids"], mask=batch["attention_mask"]
            )

        if self.omit_dialog_states:
            batch["speaker_ids"] = None

        out = self.forward(
            batch["input_ids"],
            speaker_ids=batch["speaker_ids"],
            labels=lm_labels,
            mc_labels=proj_labels,
            attention_mask=batch["attention_mask"],
        )

        if self.trp_projection_steps > 0:
            self.log("val_loss_lm", out["loss"])
            self.log("val_loss_projection", out["mc_loss"])
            total_loss = out["loss"] + out["mc_loss"]
            self.log("val_loss", total_loss)
            return {"loss": total_loss, "mc_logits": out["mc_logits"].detach(), "mc_labels": proj_labels.detach()}
        else:
            total_loss = out["loss"]
            self.log("val_loss", total_loss)
            return {"loss": total_loss}

    def validation_step_end(self, outputs):
        if self.trp_projection_steps > 0:
            shift_logits, shift_labels = self.shift_logits_labels(outputs["mc_logits"], outputs["mc_labels"])
            # print("validation_shift logits:", shift_logits.shape)
            # print("validation shit labels:", shift_labels.shape)
            self.valid_accuracy.update(shift_logits, shift_labels)
            #self.log("bAcc", bacc, prog_bar=True, logger=False)

    def validation_epoch_end(self, outputs):
        if self.trp_projection_steps > 0:
            self.log("valid_bAcc_epoch", self.valid_accuracy.compute())
            self.valid_accuracy.reset()

    def test_step(self, batch, batch_idx):
        lm_labels = self.get_labels(batch["input_ids"], mask=batch["attention_mask"])

        proj_labels = None
        if self.trp_projection_steps > 0:
            proj_labels = self.get_projection_labels(
                batch["input_ids"], mask=batch["attention_mask"]
            )

        if self.omit_dialog_states:
            batch["speaker_ids"] = None

        out = self.forward(
            batch["input_ids"],
            speaker_ids=batch["speaker_ids"],
            labels=lm_labels,
            mc_labels=proj_labels,
            attention_mask=batch["attention_mask"],
        )

        ret = self.test_generate(batch)

        if self.trp_projection_steps > 0:
            self.log("test_loss_lm", out["loss"])
            self.log("test_loss_projection", out["mc_loss"])
            total_loss = out["loss"] + out["mc_loss"]
            self.log("test_loss", total_loss)
            ret['loss'] = total_loss
            ret['mc_logits'] = out["mc_logits"].detach()
            ret['mc_labels'] = proj_labels.detach()
            return ret
        else:
            total_loss = out["loss"]
            self.log("test_loss", total_loss)
            ret['loss'] = total_loss
            return ret

    def test_step_end(self, outputs):
        for k in self.gen_scores:
            self.gen_scores[k] += outputs[k]
        
        if self.trp_projection_steps > 0:
            shift_logits, shift_labels = self.shift_logits_labels(outputs["mc_logits"], outputs["mc_labels"])
            self.test_accuracy.update(shift_logits, shift_labels)

    def test_epoch_end(self, outputs):
        for k in self.gen_scores:
            if k != 'count':
                self.log(k, self.gen_scores[k] / self.gen_scores['count'])

        if self.trp_projection_steps > 0:
            self.log("test_bAcc_epoch", self.test_accuracy.compute())
            self.test_accuracy.reset()

    @torch.no_grad()
    def test_generate(self, batch):
        input_ids = torch.numpy(batch['input_ids'], force=True)
        if not self.omit_dialog_states:
            speaker_ids = torch.numpy(batch['speaker_ids'], force=True)

        scores = {'BLEU-2': 0, 'BLEU-4': 0, 'METEOR': 0, 'NIST-2': 0, 'NIST-4': 0, 'count': 0}
        for b in range(len(input_ids)):
            # Split by eos
            split_indices = np.where(input_ids[b]==self.tokenizer.eos_token_id)[0] + 1
            input_id_ref = np.split(input_ids[b], split_indices)[:-1] # Remove padding
            
            # Generate the next turn
            for i in range(len(split_indices) - 1): # Exclude the last non-padded turn because the reference doesn't exist
                if i > 1 and split_indices[i] - split_indices[i-1] == 1: # Only for DialoGPT
                    break
                batch_for_gen = {}
                batch_for_gen["input_ids"] = torch.from_numpy(input_ids[b][:split_indices[i]]).unsqueeze(0).to(self.device)
                if self.omit_dialog_states:
                    batch_for_gen["speaker_ids"] = None
                else:
                    batch_for_gen["speaker_ids"] = torch.from_numpy(speaker_ids[b][:split_indices[i]]).unsqueeze(0).to(self.device)
                generated = generate_greedy_from_tokenized(self, batch_for_gen, 100, True)['tokens'][0]

                # Preprocess for NLTK metrics and compute them
                generated = self.tokenizer.normalize(generated)
                generated = process_for_nltk(generated, self.tokenizer)
                reference = self.tokenizer.decode(torch.from_numpy(input_id_ref[i+1]))
                reference = process_for_nltk(reference, self.tokenizer)
                sent_score = calc_score([reference], generated)
                for k, v in sent_score.items():
                    scores[k] += v
                scores['count'] += 1
            
        return scores
    
    def shift_logits_labels(self, logits, labels):
        # Shift so that tokens < n predict n
        if self.num_speakers == 2:
            shift_logits = logits[..., :-1]  # , :].contiguous()
            shift_labels = labels[..., 1:]  # .contiguous()

        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Manually select appropriate steps
        # Omit steps where label is -100 (like CrossEntropyLoss)
        indices_for_training = shift_labels != -100

        if self.num_speakers == 2:
            shift_logits = torch.masked_select(shift_logits, indices_for_training)
            shift_labels = torch.masked_select(shift_labels, indices_for_training)

        else:
            #indices_for_training_expanded = torch.reshape(indices_for_training.unsqueeze(-1).expand(shift_logits.shape), (-1, self.num_speakers))
            indices_for_training = torch.reshape(indices_for_training, (-1,))

            shift_logits = torch.reshape(shift_logits, (-1, self.num_speakers))[indices_for_training]
            shift_labels = torch.reshape(shift_labels, (-1,))[indices_for_training]

        return shift_logits, shift_labels.int()

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Specify the hyperparams for this LightningModule"""
        parser = ArgumentParser(
            parents=[parent_parser], add_help=False, conflict_handler="resolve"
        )
        parser.add_argument("--pretrained_model_name_or_path", type=str, default="gpt2")
        parser.add_argument(
            "--pretrained",
            type=bool,
            default=True,
            help="Load pretrained weights or not.",
        )

        # Model specific
        parser.add_argument("--embd_pdrob", type=float, default=None)
        parser.add_argument("--attn_pdrob", type=float, default=None)
        parser.add_argument("--resid_pdrob", type=float, default=None)
        parser.add_argument("--n_head", type=int, default=None)
        parser.add_argument("--n_layer", type=int, default=None)
        parser.add_argument("--n_embd", type=int, default=None)
        parser.add_argument("--activation_function", type=str, default=None)
        parser.add_argument("--num_speakers", type=int, default=2)

        # TurnGPT specific
        parser.add_argument(
            "--omit_dialog_states",
            action="store_true",
            help="To omit dialog-states in transformer embedding",
        )
        parser.add_argument("--trp_projection_steps", default=-1, type=int)
        parser.add_argument(
            "--no_train_first_n",
            default=-1,
            type=int,
            help="Don't train on the n first tokens.",
        )
        parser.add_argument(
            "--trp_projection_type",
            default="linear",
            type=str,
            help="'Linear' or 'Attention'",
        )

        # Training
        parser.add_argument(
            "--dropout",
            default=None,
            type=float,
            help="Set to None which uses the values in the original config.json",
        )
        parser.add_argument("--learning_rate", default=6.25e-5, type=float)
        parser.add_argument("--weight_loss", action="store_true")
        parser.add_argument("--weight_eos_token", type=float, default=1.0)
        parser.add_argument("--weight_regular_token", type=float, default=0.5)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        return parser


if __name__ == "__main__":
    from os.path import join

    parser = ArgumentParser()
    parser = TurnGPT.add_model_specific_args(parser)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # Fresh Training
    fresh = False
    if fresh:
        model = TurnGPT(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            trp_projection_steps=args.trp_projection_steps,
            trp_projection_type=args.trp_projection_type,
            weight_loss=args.weight_loss,
            weight_eos_token=args.weight_eos_token,
            weight_regular_token=args.weight_regular_token,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            pretrained=args.pretrained,
            no_train_first_n=args.no_train_first_n,
            omit_dialog_states=args.omit_dialog_states,
            weight_decay=args.weight_decay
        )
        model.init_tokenizer()
        model.initialize_special_embeddings()
    else:
        # projection
        chpt = join(
            "assets/TurnGPT_proj/version_0/checkpoints/epoch=45-val_loss=-3.37196.ckpt"
        )
        # only LM
        chpt = join(
            "assets/TurnGPT/version_0/checkpoints/epoch=11-val_loss=1.23640.ckpt"
        )
        model = TurnGPT.load_from_checkpoint(chpt).to("cuda")

    turn_list = [
        "Hello there I basically had the worst day of my life",
        "Oh no, what happened?",
        "Do you want the long or the short story?",
    ]

    gen = generate(
        model,
        context=turn_list,
        n_steps=200,
        top_p=0.9,
        top_k=-1,
        n_trajectories=20,
        strategy="sample",
        stop_at_eos=True,
    )
    # remove duplicates
    l = (gen["input_ids"][0] != -1).sum()
    G = {"tokens": [gen["tokens"][0]], "probs": [gen["probs"][0][:l].cpu()]}
    for i, g in enumerate(gen["tokens"][1:]):
        if g not in G["tokens"]:
            l = (gen["input_ids"][i] != -1).sum()
            G["tokens"].append(g)
            G["probs"].append(gen["probs"][i][:l].cpu())

    #########################################################
    turn_list = [
        [
            "Hello there I basically had the worst day of my life",
            "Oh no, what happened?",
            "Do you want the long or the short story?",
        ],
        ["yesterday we met in the park", "okay when will you meet again", "tomorrow"],
    ]
    out = model.string_list_to_trp(turn_list)
    # for k, v in out.items():
    #     print(f"{k}: {v}")

    for b in range(out["trp_probs"].shape[0]):
        proj = out["trp_proj"][b].cpu() if "trp_proj" in out else None
        fig, ax = plot_trp(
            trp=out["trp_probs"][b].cpu(),
            proj=proj,
            text=out["tokens"][b],
            unk_token=model.tokenizer._tokenizer.unk_token,
            plot=True,
        )
