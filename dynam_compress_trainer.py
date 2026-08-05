import os 
import sys 
from copy import deepcopy 
from typing import Any, Dict, List, Tuple, Optional
from argparse import Namespace
from dataclasses import dataclass
import time
import contextlib
import functools

import torch 
from torch import nn 
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import matplotlib.pyplot as plt


from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.optimization import GreedyLR
from transformers import (
    PreTrainedModel, 
)
from transformers.trainer_utils import (
    SaveStrategy,
    has_length, 
    EvalLoopOutput,
    EvalPrediction, 
    denumpify_detensorize
)
from transformers.trainer_pt_utils import (
    nested_gather,
    EvalLoopContainer,
    find_batch_size,
    IterableDatasetShard,
)
from accelerate import skip_first_batches
from accelerate.utils.memory import clear_device_cache
from accelerate.utils import (
    DataLoaderConfiguration,
    DistributedDataParallelKwargs,
    DistributedType,
    GradientAccumulationPlugin,
    load_fsdp_model,
    load_fsdp_optimizer,
    release_memory,
    save_fsdp_model,
    save_fsdp_optimizer,
)
from trainer import Trainer, logger
from models.compression_utils import compute_soft_chunk_mask

class ChunkerTrainer(Trainer): 
    def __init__(self, 
                 *args, 
                 **kwargs): 
        super().__init__(*args, **kwargs) 
        self.processing_class.padding_side = "left"

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, torch.Tensor | Any]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                clear_device_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                loss = loss / self.current_gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, torch.Tensor | Any]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If not passed, the loss is computed
                using the default batch size reduction logic.

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculation might be slightly inaccurate when performing gradient accumulation.
        """

        pass
        labels = inputs.pop("labels")
        inputs["labels"] = inputs["token_type_ids"] 
        outputs = model(**inputs)
        loss = outputs.loss 

        probs = F.sigmoid(outputs.logits) 

        # loss += 0.05 * ((probs.sum(dim=(-1, -2)) - inputs["labels"].sum(dim=1))**2).mean()
        ctx_start = inputs["attention_mask"]
        compression_rates = []
        for i in range(probs.shape[0]): 
            ctx_start = (inputs["attention_mask"][i] == 1).nonzero(as_tuple=False)[0].item()
            ctx_end = (inputs["token_type_ids"][i] == 0).nonzero(as_tuple=False)[-1].item() + 1
            # ctx_end = inputs["attention_mask"][i].shape[0] - ans_len 
            comp_mass = probs[i][ctx_start:ctx_end, 0].sum().item()
            total_mass = ctx_end - ctx_start
            compression_rates.append(total_mass / comp_mass)
        mean_comp_rate = torch.tensor(compression_rates, device=self.accelerator.device).mean()

        gathered_comp_rate = self.accelerator.gather_for_metrics(mean_comp_rate)
        self.state.comp_rate = gathered_comp_rate.mean().detach().cpu().item() 

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix = "eval"
    ):
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only


        model = self._wrap_model(self.model, training=False)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and not self.args.torch_compile)
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_comp_rates = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size


            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )


            chunker_inputs = deepcopy(inputs) 
            chunker_inputs["labels"] = chunker_inputs["token_type_ids"]
            chunker_inputs.pop("token_type_ids") 
            chunker_outputs = self.model(**chunker_inputs)
            chunker_logits = chunker_outputs["logits"] 
            chunk_probs = F.sigmoid(chunker_logits) 
            compression_rates = [] 
            for i in range(chunk_probs.shape[0]): 
                ctx_start = (chunker_inputs["attention_mask"][i] == 1).nonzero(as_tuple=False)[0].item()
                ctx_end = (inputs["token_type_ids"][i] == 0).nonzero(as_tuple=False)[-1].item() + 1
                comp_mass = chunk_probs[i][ctx_start:ctx_end, 0].sum().item()
                total_mass = ctx_end - ctx_start
                compression_rates.append(total_mass / comp_mass)
            mean_comp_rate = torch.tensor(compression_rates, device=self.accelerator.device).mean()

            mean_comp_rate = self.gather_function(mean_comp_rate.repeat(batch_size))
            all_comp_rates.add(mean_comp_rate)


            if step == 0: 
                with torch.no_grad(): 
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    if self.accelerator.is_main_process:
                        probs = 1/(1+torch.exp(-outputs.logits))
                        softmask = compute_soft_chunk_mask(probs, probs.shape[1], probs.shape[1])
                        plt.imshow(softmask.detach().cpu().numpy()[0, 0])
                        plt.savefig(os.path.join(self.args.output_dir, f"softmask-{self.state.global_step}.png"),dpi=300)


            # Update containers
            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()
                all_comp_rates.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()
        all_comp_rates = all_comp_rates.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time


        metrics["compression_rate"] = all_comp_rates.mean().item()


        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


    def _maybe_log_save_evaluate(
        self,
        tr_loss: torch.Tensor,
        grad_norm: torch.Tensor | float | None,
        model: nn.Module,
        trial: "optuna.Trial | dict[str, Any] | None",
        epoch: float,
        ignore_keys_for_eval: list[str] | None,
        start_time: float,
        learning_rate: float | None = None,
    ) -> None:
        """Log metrics, run evaluation, and save checkpoints if the current training state requires it."""
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            # if is_torch_xla_available():
            #     xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = nested_gather(tr_loss, self.args.parallel_mode).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            # logs["compression_rate"] = self.data_collator.rate.item()
            # logs["chunk_diff_norm"] = self.data_collator.get_chunk_diff()
            # logs["uniform_chunk_diff_norm"] = self.data_collator.get_uniform_chunk_diff()
            # logs["multi_token_chunk_diff_norm"] = self.data_collator.get_multi_tok_chunk_diff()
            logs["compression_rate"] = getattr(self.state, "comp_rate", None)

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

            best_tfmr_path = os.path.join(self.args.output_dir, "best_tfmr")
            if self.is_fsdp_enabled:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(
                        best_tfmr_path,
                        is_main_process=self.accelerator.is_main_process,
                        save_function=self.accelerator.save,
                        state_dict=self.accelerator.get_state_dict(model))
                self.processing_class.save_pretrained(best_tfmr_path)
            elif self.accelerator.unwrap_model(self.model) == self.model:
                self.model.save_pretrained(best_tfmr_path)
                self.processing_class.save_pretrained(best_tfmr_path)
            elif isinstance(self.model, DDP): 
                chunking_model = self.model.module 
                chunking_model.save_pretrained(best_tfmr_path)
            else: 
                raise NotImplementedError


    # def create_optimizer(self, model=None) -> torch.optim.Optimizer:
    #     """
    #     Setup the optimizer.

    #     We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    #     Trainer's init through `optimizers`, or subclass and override this method in a subclass.

    #     Returns:
    #         `torch.optim.Optimizer`: The optimizer instance.
    #     """
    #     opt_model = self.model if model is None else model
    #     if isinstance(opt_model, DDP): 
    #         opt_model = opt_model.module

    #     if self.optimizer is None:
    #         decay_parameters = self.get_decay_parameter_names(opt_model)
    #         optimizer_grouped_parameters = [
    #             {
    #                 "params": opt_model.classifier.parameters(),
    #                 "weight_decay": 0,
    #                 "lr": self.args.learning_rate, 
    #             },
    #             {
    #                 "params": opt_model.model.layers[-1].parameters(), 
    #                 "weight_decay": 0.0,
    #                 "lr": self.args.learning_rate * 0
    #             },
    #         ]

    #         if self.optimizer_cls_and_kwargs is not None:
    #             optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
    #         else:
    #             optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

    #         # Check if this is a factory (for complex optimizers like Muon, Dion)
    #         # Factories are instantiated first, then called with (opt_model, **kwargs)
    #         if False: #is_optimizer_factory(optimizer_cls):
    #             self.optimizer = optimizer_cls()(opt_model, **optimizer_kwargs)
    #         else:
    #             # Standard optimizer class instantiation
    #             # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
    #             # e.g. for GaLore optimizer.
    #             if "params" in optimizer_kwargs:
    #                 optimizer_grouped_parameters = optimizer_kwargs.pop("params")

    #             # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
    #             # e.g. for LOMO optimizer.
    #             if "model" in optimizer_kwargs:
    #                 optimizer_grouped_parameters = optimizer_kwargs.pop("model")

    #             # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
    #             # to avoid arguments conflicts.
    #             if "optimizer_dict" in optimizer_kwargs:
    #                 optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

    #             self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


    #     return self.optimizer



@dataclass
class CompressTrainingArguments(TrainingArguments): 
    def __init__(self, 
                 mask_mode: str = "soft", 
                 alpha_unif: float | None = None, 
                 chunk_lr: float | None = None, 
                 *args, **kwargs): 
        self.mask_mode = mask_mode 
        self.alpha_unif = alpha_unif if alpha_unif is not None else 0
        self.chunk_lr = chunk_lr if chunk_lr is not None else 0
        super().__init__(*args, **kwargs)




class CompressTrainer(Trainer): 
    def __init__(self, 
                 chunking_model: PreTrainedModel, 
                 *args, **kwargs): 
        self.chunking_model = chunking_model 
        self.mask_mode = kwargs["args"].mask_mode
        self.alpha_unif = kwargs["args"].alpha_unif
        super().__init__(*args, **kwargs)
        self.processing_class.padding_side = "left"
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None, 
    ) -> torch.Tensor | tuple[torch.Tensor, Any]: 
        reg_loss = torch.tensor(0, device=self.chunking_model.device)
        if self.mask_mode == "hard": 
            chunk_signal = F.one_hot(inputs["token_type_ids"], num_classes=3)
        elif self.mask_mode == "soft":
            chunker_inputs = deepcopy(inputs) 
            chunker_inputs["labels"] = inputs["token_type_ids"]
            chunker_inputs.pop("token_type_ids")
            chunker_outputs = self.chunking_model(**chunker_inputs)

            logits = chunker_outputs["logits"] 
            reg_loss = chunker_outputs["loss"]

            chunk_signal = 1/(1 + torch.exp(-logits))
            # chunk_signal.retain_grad()

            comp_count = ((chunker_inputs["labels"] == 1).int()).sum(dim=-1)
            comp_mass = chunk_signal.sum(dim=(-1, -2)) 
            reg_loss += ((comp_count - comp_mass)**2).mean()
            self.accelerator.wait_for_everyone()
        elif self.mask_mode == "contextless": 
            modded_ids = torch.where(inputs["token_type_ids"] == 2, inputs["token_type_ids"], 0)
            chunk_signal = F.one_hot(modded_ids, num_classes=3)
        elif self.mask_mode == "full": 
            chunk_signal = None 

        inputs["chunk_signal"] = chunk_signal
        pass
        # torch.autograd.set_detect_anomaly(True)

        outputs = model(**inputs)

        # User-defined compute_loss function
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        self.state.regularization_loss = reg_loss.detach().cpu().item() 
        self.state.ce_loss = loss.detach().cpu().item()

        train_mode = torch.is_grad_enabled()
        if train_mode: 
            loss += self.alpha_unif * reg_loss

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        return (loss, outputs) if return_outputs else loss 

    def _prepare_for_training(self, max_steps, train_dataloader, resume_from_checkpoint):
        model, train_dataloader = super()._prepare_for_training(max_steps, train_dataloader, resume_from_checkpoint)
        self.chunking_model = self.chunking_model.to(self.accelerator.device)
        if self.accelerator.num_processes > 1: 
            self.chunking_model = DDP(self.chunking_model, device_ids=[self.accelerator.device.index])
        chunk_optim_cls, chunk_optim_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.chunking_model)
        chunk_optim_kwargs["lr"] = self.args.chunk_lr
        self.chunk_optim = chunk_optim_cls(params=self.chunking_model.parameters(), **chunk_optim_kwargs)
        self.accelerator.wait_for_everyone()
        return model, train_dataloader
    
    def _maybe_log_save_evaluate(
        self,
        tr_loss: torch.Tensor,
        grad_norm: torch.Tensor | float | None,
        model: nn.Module,
        trial: "optuna.Trial | dict[str, Any] | None",
        epoch: float,
        ignore_keys_for_eval: list[str] | None,
        start_time: float,
        learning_rate: float | None = None,
    ) -> None:
        """Log metrics, run evaluation, and save checkpoints if the current training state requires it."""
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            # if is_torch_xla_available():
            #     xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = nested_gather(tr_loss, self.args.parallel_mode).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            # logs["compression_rate"] = self.data_collator.rate.item()
            # logs["chunk_diff_norm"] = self.data_collator.get_chunk_diff()
            # logs["uniform_chunk_diff_norm"] = self.data_collator.get_uniform_chunk_diff()
            # logs["multi_token_chunk_diff_norm"] = self.data_collator.get_multi_tok_chunk_diff()
            logs["ce_loss"] = getattr(self.state, "ce_loss", None)
            logs["reg_loss"] = getattr(self.state, "regularization_loss", None)

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

            best_tfmr_path = os.path.join(self.args.output_dir, "best_tfmr")
            best_chunker_path = os.path.join(self.args.output_dir, "best_chunker")
            if self.is_fsdp_enabled:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(
                        best_tfmr_path,
                        is_main_process=self.accelerator.is_main_process,
                        save_function=self.accelerator.save,
                        state_dict=self.accelerator.get_state_dict(model))
                self.processing_class.save_pretrained(best_tfmr_path)

            elif self.accelerator.unwrap_model(self.model) == self.model:
                self.model.save_pretrained(best_tfmr_path)
                self.processing_class.save_pretrained(best_tfmr_path)
            else:
                raise NotImplementedError("Implement saving for distributed model")

            if self.chunking_model is not None:
                if isinstance(self.chunking_model, DDP): 
                    chunking_model = self.chunking_model.module 
                else: 
                    chunking_model = self.chunking_model 
                chunking_model.save_pretrained(best_chunker_path)


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(self.model, training=False)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and not self.args.torch_compile)
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_comp_rates = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )
            if self.chunking_model is not None:
                chunker_inputs = deepcopy(inputs) 
                chunker_inputs["labels"] = chunker_inputs["token_type_ids"]
                chunker_inputs.pop("token_type_ids") 
                chunker_outputs = self.chunking_model(**chunker_inputs)
                chunker_logits = chunker_outputs["logits"] 
                chunk_probs = F.sigmoid(chunker_logits) 
                compression_rates = [] 
                for i in range(chunk_probs.shape[0]): 
                    ctx_start = (chunker_inputs["attention_mask"][i] == 1).nonzero(as_tuple=False)[0].item()
                    ctx_end = (inputs["token_type_ids"][i] == 0).nonzero(as_tuple=False)[-1].item() + 1
                    comp_mass = chunk_probs[i][ctx_start:ctx_end, 0].sum().item()
                    total_mass = ctx_end - ctx_start
                    compression_rates.append(total_mass / comp_mass)
                mean_comp_rate = torch.tensor(compression_rates, device=self.accelerator.device).mean()

                mean_comp_rate = self.gather_function(mean_comp_rate.repeat(batch_size))
                all_comp_rates.add(mean_comp_rate)

                if step == 0: 
                    softmask = compute_soft_chunk_mask(chunk_probs, chunk_probs.shape[1], chunk_probs.shape[1])
                    softmask_ = softmask.detach().cpu().numpy()[0, 0]
                    plt.imshow(softmask.detach().cpu().numpy()[0, 0])
                    plt.title(f"Max: {softmask_.max()}, Min: {softmask_.min()}")
                    plt.savefig(os.path.join(self.args.output_dir, f"softmask-{self.state.global_step}.png"),dpi=300)
                    pass


            # Update containers
            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()
                all_comp_rates.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()
        all_comp_rates = all_comp_rates.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        metrics["compression_rate"] = all_comp_rates.mean().item()


        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


    def _run_epoch(
        self,
        model,
        epoch,
        train_dataloader,
        steps_in_epoch,
        num_update_steps_per_epoch,
        trial,
        ignore_keys_for_eval,
        start_time,
        resume_from_checkpoint,
        epochs_trained,
        steps_trained_in_current_epoch,
    ):
        """Run one full pass over the dataloader."""

        step = -1
        grad_norm = None
        learning_rate = None
        rng_to_sync = False

        # Handle resumption from checkpoint: skip already-trained batches in the resumed epoch
        num_update_steps_trained = 0
        if epoch == epochs_trained and resume_from_checkpoint is not None:
            if steps_trained_in_current_epoch > 0 and not self.args.ignore_data_skip:
                train_dataloader = skip_first_batches(train_dataloader, steps_trained_in_current_epoch)
                step = steps_trained_in_current_epoch - 1
                num_update_steps_trained = steps_trained_in_current_epoch // self.args.gradient_accumulation_steps
                rng_to_sync = True
            elif steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)
        epoch_iterator = iter(train_dataloader)

        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = steps_in_epoch % self.args.gradient_accumulation_steps
        if remainder == 0:
            remainder = self.args.gradient_accumulation_steps

        # Outer loop: one iteration per optimizer step. Each iteration prefetches
        # `gradient_accumulation_steps` batches (fewer for the last step if the epoch
        # doesn't divide evenly).
        for update_step in range(num_update_steps_trained, num_update_steps_per_epoch):
            num_batches = (
                self.args.gradient_accumulation_steps if update_step != (num_update_steps_per_epoch - 1) else remainder
            )
            batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, self.args.device)

            # This is used to correctly scale the loss when the last accumulation step has fewer batches.
            # Not used if `num_items_in_batch` is not None.
            self.current_gradient_accumulation_steps = len(batch_samples)

            # need to sync after if we skipped the batches in `get_batch_samples` for shuffle order reason
            if rng_to_sync:
                self._load_rng_state(resume_from_checkpoint)
                rng_to_sync = False

            # Inner loop: forward + backward for each micro-batch. Gradients are
            # accumulated without syncing until the last micro-batch, then we clip,
            # step the optimizer, and log/save/evaluate.
            for i, inputs in enumerate(batch_samples):
                step += 1
                do_sync_step = (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                # Since we perform prefetching, we need to manually set sync_gradients
                self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                if step % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                # We sync the gradients in the following cases: 1. sync_each_batch set to True 2. Using deepspeed 3. when we are at the last batch sample
                if (
                    self.accelerator.gradient_state.plugin_kwargs.get("sync_each_batch", False)
                    or self.accelerator.distributed_type == DistributedType.DEEPSPEED
                    or i == len(batch_samples) - 1
                ):
                    sync_context = contextlib.nullcontext
                else:
                    sync_context = functools.partial(self.accelerator.no_sync, model=model)
                with sync_context():
                    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                if (
                    self.args.logging_nan_inf_filter
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    self._tr_loss += self._tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if self._tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {self._tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    self._tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))
                self._track_num_input_tokens(inputs)

                if do_sync_step:
                    grad_norm = None
                    if self.args.max_grad_norm > 0:
                        grad_norm = self._clip_grad_norm(model)
                    grad_norm = self._get_grad_norm(model, grad_norm=grad_norm)

                    self.control = self.callback_handler.on_pre_optimizer_step(self.args, self.state, self.control)
                    self.optimizer.step()
                    self.control = self.callback_handler.on_optimizer_step(self.args, self.state, self.control)

                    if self.chunking_model is not None: 
                        self.chunk_optim.step() 
                        self.chunk_optim.zero_grad() 


                    # get leaning rate before update
                    learning_rate = self._get_learning_rate()

                    if not self.accelerator.optimizer_step_was_skipped:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, GreedyLR)):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                    self._maybe_log_save_evaluate(
                        self._tr_loss,
                        grad_norm,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                        learning_rate=learning_rate,
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(self.args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        # PyTorch/XLA relies on the dataloader to insert mark_step each iteration.
        # When we break out of the loop early, we flush the pending graph manually.

        if step < 0:
            logger.warning(
                "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                f" num_steps ({self.state.max_steps}) higher than the number of available samples."
            )
            self.control.should_training_stop = True

        self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
        self._maybe_log_save_evaluate(
            self._tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate=learning_rate,
        )