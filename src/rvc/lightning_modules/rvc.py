from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from pytorch_lightning import LightningModule

from ..torch_modules.discriminator import MultiPeriodDiscriminator
from ..torch_modules.generator import SynthesizerTrnMs256NSFsid_nono

class RVC(LightningModule):
    def __init__(
        self, 
        d_lr: float = 1e-4,
        g_lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.generator = SynthesizerTrnMs256NSFsid_nono()
        self.discriminator = MultiPeriodDiscriminator(use_spectral_norm=True)

    def configure_optimizers(self):
        optimizer_g = AdamW(self.generator.parameters(), lr=self.g_lr)
        optimizer_d = AdamW(self.discriminator.parameters(), lr=self.d_lr)
        scheduler_g = ExponentialLR(optimizer_g, gamma=hps.train.lr_decay)  # NOTE last_epoch=epoch_str - 2
        scheduler_d = ExponentialLR(optimizer_d, gamma=hps.train.lr_decay)
        return (
            {
                "optimizer": optimizer_g,
                "lr_scheduler": {
                    "scheduler": scheduler_g,
                    "monitor": "metric_to_track",
                },
            },
            {
                "optimizer": optimizer_d, 
                "lr_scheduler": {
                    "scheduler": scheduler_d,
                    "monitor": "metric_to_track",
                },
            }
        )
