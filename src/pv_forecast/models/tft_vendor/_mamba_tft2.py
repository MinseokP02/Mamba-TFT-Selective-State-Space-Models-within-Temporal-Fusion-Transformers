"""Mamba-TFT with independent history encoder and causal future decoder."""

import torch
from torch import nn
from pytorch_forecasting.utils import create_mask

from ._mamba_tft import Mamba, MambaTemporalFusionTransformer
from .sub_modules import AddNorm, GatedLinearUnit


class SeparateMambaTemporalFusionTransformer(MambaTemporalFusionTransformer):
    """TFT variant with separate Mamba modules for encoder and decoder.

    The encoder processes historical observations only. The decoder processes
    future-known covariates causally and is conditioned on the final valid
    encoder representation. TFT attention after these blocks is unchanged.
    """

    def __init__(
        self,
        encoder_mamba_layers: int | None = None,
        encoder_mamba_d_state: int | None = None,
        encoder_mamba_d_conv: int | None = None,
        encoder_mamba_expand: int | None = None,
        decoder_mamba_layers: int | None = None,
        decoder_mamba_d_state: int | None = None,
        decoder_mamba_d_conv: int | None = None,
        decoder_mamba_expand: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if Mamba is None:
            raise ImportError("SeparateMambaTemporalFusionTransformer requires mamba-ssm")

        def value(explicit, fallback):
            return fallback if explicit is None else explicit

        common_layers = int(self.hparams.mamba_layers)
        common_state = int(self.hparams.mamba_d_state)
        common_conv = int(self.hparams.mamba_d_conv)
        common_expand = int(self.hparams.mamba_expand)

        enc_layers = int(value(encoder_mamba_layers, common_layers))
        enc_state = int(value(encoder_mamba_d_state, common_state))
        enc_conv = int(value(encoder_mamba_d_conv, common_conv))
        enc_expand = int(value(encoder_mamba_expand, common_expand))
        dec_layers = int(value(decoder_mamba_layers, common_layers))
        dec_state = int(value(decoder_mamba_d_state, common_state))
        dec_conv = int(value(decoder_mamba_d_conv, common_conv))
        dec_expand = int(value(decoder_mamba_expand, common_expand))

        # Remove the original joint backbone so it is not optimized or saved.
        del self.mamba_blocks
        del self.mamba_norms
        del self.mamba_dropout

        hidden_size = int(self.hparams.hidden_size)
        dropout = float(self.hparams.dropout)
        self.encoder_mamba_blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=hidden_size,
                    d_state=enc_state,
                    d_conv=enc_conv,
                    expand=enc_expand,
                )
                for _ in range(enc_layers)
            ]
        )
        self.encoder_mamba_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(enc_layers)]
        )
        self.decoder_mamba_blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=hidden_size,
                    d_state=dec_state,
                    d_conv=dec_conv,
                    expand=dec_expand,
                )
                for _ in range(dec_layers)
            ]
        )
        self.decoder_mamba_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_size) for _ in range(dec_layers)]
        )
        self.encoder_mamba_dropout = nn.Dropout(dropout)
        self.decoder_mamba_dropout = nn.Dropout(dropout)

        # Learned analogue of passing an encoder hidden state to a decoder.
        self.encoder_to_decoder = nn.Linear(hidden_size, hidden_size)

        # Encoder and decoder distributions differ, so do not share these.
        self.post_temporal_gate_encoder = GatedLinearUnit(hidden_size, dropout=dropout)
        self.post_temporal_gate_decoder = GatedLinearUnit(hidden_size, dropout=dropout)
        self.post_temporal_add_norm_encoder = AddNorm(hidden_size, trainable_add=False)
        self.post_temporal_add_norm_decoder = AddNorm(hidden_size, trainable_add=False)

        self.save_hyperparameters(
            {
                "encoder_mamba_layers": enc_layers,
                "encoder_mamba_d_state": enc_state,
                "encoder_mamba_d_conv": enc_conv,
                "encoder_mamba_expand": enc_expand,
                "decoder_mamba_layers": dec_layers,
                "decoder_mamba_d_state": dec_state,
                "decoder_mamba_d_conv": dec_conv,
                "decoder_mamba_expand": dec_expand,
            }
        )

    @staticmethod
    def _apply_stack(x, blocks, norms, dropout, padding_mask):
        output = x.masked_fill(padding_mask, 0.0)
        for block, norm in zip(blocks, norms):
            residual = output
            output = block(output)
            output = dropout(output)
            output = norm(output + residual)
            output = output.masked_fill(padding_mask, 0.0)
        return output

    def _run_temporal_blocks(
        self,
        embeddings_varying_encoder: torch.Tensor,
        embeddings_varying_decoder: torch.Tensor,
        encoder_lengths: torch.LongTensor,
        decoder_lengths: torch.LongTensor,
        static_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_steps = embeddings_varying_encoder.size(1)
        decoder_steps = embeddings_varying_decoder.size(1)

        encoder_input = self.static_context_mamba(
            embeddings_varying_encoder,
            self.expand_static_context(static_embedding, encoder_steps),
        )
        decoder_input = self.static_context_mamba(
            embeddings_varying_decoder,
            self.expand_static_context(static_embedding, decoder_steps),
        )
        encoder_mask = create_mask(encoder_steps, encoder_lengths).unsqueeze(-1)
        decoder_mask = create_mask(decoder_steps, decoder_lengths).unsqueeze(-1)

        encoder_output = self._apply_stack(
            encoder_input,
            self.encoder_mamba_blocks,
            self.encoder_mamba_norms,
            self.encoder_mamba_dropout,
            encoder_mask,
        )

        # Gather the final valid history state and condition every decoder step.
        last_index = (encoder_lengths - 1).clamp_min(0)
        batch_index = torch.arange(encoder_output.size(0), device=encoder_output.device)
        encoder_context = encoder_output[batch_index, last_index]
        decoder_input = decoder_input + self.encoder_to_decoder(encoder_context).unsqueeze(1)

        decoder_output = self._apply_stack(
            decoder_input,
            self.decoder_mamba_blocks,
            self.decoder_mamba_norms,
            self.decoder_mamba_dropout,
            decoder_mask,
        )
        return encoder_output, decoder_output
