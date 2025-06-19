import os
import logging
import torch
from nemo.lightning.pytorch.strategies import MegatronStrategy
from lightning.pytorch.trainer.states import TrainerFn


class CustomMegatronStrategy(MegatronStrategy):
    def selective_restore(self) -> None:
        """Implements selective restoration of checkpoint"""
        if not self.restore_config:
            return

        logging.info(f"Doing selective restore from {self.restore_config}")

        checkpoint = self.load_checkpoint(checkpoint_path=self.restore_config.path, selective_restore=True)

        if self.restore_config.load_model_state:
            logging.info(f"Restoring model weights from {self.restore_config}")
            strict = True if self.ckpt_load_strictness is None else self.ckpt_load_strictness
            self.load_model_state_dict(checkpoint=checkpoint, strict=strict)

        if self.restore_config.load_optim_state:
            logging.info(f"Restoring optimizer states from {self.restore_config}")
            self.load_optimizer_state_dict(checkpoint=checkpoint, selective_restore=True)
        
        fit_loop = self.trainer.fit_loop
        assert self.trainer.state.fn is not None
        state_dict = torch.load(os.path.join(self.restore_config.path, "weights", "common.pt"))
        if state_dict is not None:
            if self.trainer.state.fn == TrainerFn.FITTING:
                fit_loop.load_state_dict(state_dict['loops']["fit_loop"])
            elif self.trainer.state.fn == TrainerFn.VALIDATING:
                self.trainer.validate_loop.load_state_dict(state_dict['loops']["validate_loop"])
            elif self.trainer.state.fn == TrainerFn.TESTING:
                self.trainer.test_loop.load_state_dict(state_dict['loops']["test_loop"])
            elif self.trainer.state.fn == TrainerFn.PREDICTING:
                self.trainer.predict_loop.load_state_dict(state_dict['loops']["predict_loop"])
        
        # self.model.optim.lr_scheduler.load_state_dict(state_dict["lr_schedulers"][0])
        
        logging.info(f"Finished restoring from {self.restore_config}, cleaning up.")
        torch.cuda.empty_cache()
        # wait for all to catch up
        self.trainer.strategy.barrier("MegatronStrategy.restore_end")