import pytorch_lightning as pl

class FreezeDecoderCallback(pl.Callback):
    # For freezing decoder when training for de novo task.
    def __init__(self, freeze_epochs: int = 3):
        self.freeze_epochs = freeze_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch < self.freeze_epochs:
            for param in pl_module.decoder_embed.parameters():
                param.requires_grad = False
            for param in pl_module.transformer_decoder.parameters():
                param.requires_grad = False
            for param in pl_module.decoder_fc.parameters():
                param.requires_grad = False
            for param in pl_module.pos_encoder.parameters():
                param.requires_grad = False
            print(f"Epoch {trainer.current_epoch}: Pretrained decoder frozen.")
        else:
            for param in pl_module.decoder_embed.parameters():
                param.requires_grad = True
            for param in pl_module.transformer_decoder.parameters():
                param.requires_grad = True
            for param in pl_module.decoder_fc.parameters():
                param.requires_grad = True
            for param in pl_module.pos_encoder.parameters():
                param.requires_grad = True
            print(f"Epoch {trainer.current_epoch}: Pretrained decoder unfrozen.")