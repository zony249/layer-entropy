import os 
from typing import List, Dict, Union, Any, Optional, Tuple

import torch 
from torch import nn

from transformers.utils import (
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    logging,
)

from accelerate import (
    Accelerator
)

from transformers.trainer_utils import (
    SaveStrategy,
)

from transformers.trainer_pt_utils import (
    nested_detach,
)

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

if is_datasets_available():
    import datasets


logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

from trainer import Trainer



class GistTrainer(Trainer): 
    def __init__(self, 
                 generate_kwargs: Optional[dict] = None, 
                *args, 
                **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_kwargs = generate_kwargs if generate_kwargs is not None else {
            "max_new_tokens": 128, 
            "do_sample": True,
            "top_p": 0.95, 
            "top_k": 20, 
            "temperature": 0.6, 
            "num_beams": 1,  
            "use_cache": True
        }
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None):
        
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            # if is_torch_xla_available():
            #     xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

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
            if self.accelerator.unwrap_model(self.model) == self.model: 
                self.model.save_pretrained(best_tfmr_path) 
                self.processing_class.save_pretrained(best_tfmr_path)
            else: 
                raise NotImplementedError("Implement saving for distributed model")


    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss")
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = len(self.label_names) == 0 and return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                # raw_outputs = smp_forward_only(model, inputs)
                # if has_labels or loss_without_labels:
                #     if isinstance(raw_outputs, dict):
                #         loss_mb = raw_outputs["loss"]
                #         logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                #     else:
                #         loss_mb = raw_outputs[0]
                #         logits_mb = raw_outputs[1:]

                #     loss = loss_mb.reduce_mean().detach().cpu()
                #     logits = smp_nested_concat(logits_mb)
                # else:
                #     loss = None
                #     if isinstance(raw_outputs, dict):
                #         logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                #     else:
                #         logits_mb = raw_outputs
                #     logits = smp_nested_concat(logits_mb)
                pass
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        num_items_in_batch = self._get_num_items_in_batch([inputs], self.args.device)
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                        )
                    loss = loss.detach().mean()

                    # logits = self.generate(model, inputs, self.generate_kwargs)


                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels) 

    def generate(self, 
                model, 
                inputs, 
                generate_kwargs):
        model.eval() 
        inputs_filtered = self.filter_out_answers(inputs, match_with="Answer:")
        for k, v in inputs_filtered.items():
            generate_kwargs[k] = v
        outputs = model.generate(**generate_kwargs) 
        
        outputs[:, :inputs_filtered["input_ids"].shape[1]] = -100

        return outputs

    def filter_out_answers(self, 
                           inputs: Dict[str, torch.Tensor], 
                           match_with: Optional[str] = "Answer:"):
        text_input = self.processing_class.batch_decode(inputs["input_ids"])
        stripped_text_input = []
        for text in text_input: 
            idx = text.find(match_with) + len(match_with)
            if idx >= 0: 
                stripped_text = text[:idx]
                stripped_text_input.append(stripped_text)

        inputs = self.processing_class(stripped_text_input, 
                                       padding=True,
                                       padding_side="left",  
                                       return_tensors="pt")
        inputs = inputs.to(self.accelerator.device)
        return inputs
