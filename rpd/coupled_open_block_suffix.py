"""Sparse open-window wrapper for the coupled suffix score."""
from dataclasses import fields
from types import SimpleNamespace

import torch

from .coupled_gated_suffix import (CoupledGatedSuffixReadout,
                                   CoupledGatedSuffixScore,
                                   CoupledGatedSuffixTokens)


class CoupledOpenBlockReadout(CoupledGatedSuffixReadout):
    @torch.inference_mode()
    def read(self, raw_final_logits, indices):
        if not self.active:
            raise RuntimeError('begin and run one model forward before reading')
        if indices.ndim != 1 or indices.numel() == 0:
            raise ValueError('read expects nonempty one-dimensional row indices')
        names = ('norm', 'head', 'logit_scale', 'layers', 'kappa', 'tau',
                 'd', 'canvas_shape')
        view = SimpleNamespace(**{name: getattr(self, name) for name in names})
        view.active = True
        view.positions = self.positions[indices]
        view.batch = self.batch[indices]
        view.source_position = self.source_position[indices]
        view.hidden = {layer: hidden[indices]
                       for layer, hidden in self.hidden.items()}
        return CoupledGatedSuffixReadout.finish(view, raw_final_logits)

    def end(self):
        self.hidden.clear()
        self.active = False
