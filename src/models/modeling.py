import torch
import copy

import clip.clip as clip

from src.models import utils
import open_clip
from six import add_metaclass
import torch.nn as nn
from contextlib import contextmanager
import logging


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        # self.model, self.train_preprocess, self.val_preprocess = clip.load(
        #     args.model, args.device, jit=False)

        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained=args.pretrained, device=args.device, jit=False)
        
        self.cache_dir = args.cache_dir

        print("Has transformer: ", hasattr(self.model, 'transformer'))
        if not keep_lang and hasattr(self.model, 'transformer'):
            print('Removing language encoder')
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def save(self, filename):
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return utils.torch_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def save(self, filename):
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)

# class PatchModules(type):
#     def __call__(cls, state, *args, **kwargs):
#         r"""Called when you call ReparamModule(...) """
#         net = type.__call__(cls, state, *args, **kwargs)

#         # collect weight (module, name) pairs
#         # flatten weights
#         w_modules_names = []

#         for m in net.modules():
#             for n, p in m.named_parameters(recurse=False):
#                 if p is not None:
#                     w_modules_names.append((m, n))
#             for n, b in m.named_buffers(recurse=False):
#                 if b is not None:
#                     logging.warning((
#                         '{} contains buffer {}. The buffer will be treated as '
#                         'a constant and assumed not to change during gradient '
#                         'steps. If this assumption is violated (e.g., '
#                         'BatchNorm*d\'s running_mean/var), the computation will '
#                         'be incorrect.').format(m.__class__.__name__, n))

#         net._weights_module_names = tuple(w_modules_names)

#         # Put to correct device before we do stuff on parameters
#         # net = net.to(state.device)
#         net = net.to("cuda")
#         ws = tuple(m._parameters[n].detach() for m, n in w_modules_names)

#         assert len(set(w.dtype for w in ws)) == 1

#         # reparam to a single flat parameter
#         net._weights_numels = tuple(w.numel() for w in ws)
#         net._weights_shapes = tuple(w.shape for w in ws)
#         with torch.no_grad():
#             flat_w = torch.cat([w.reshape(-1) for w in ws], 0)

#         # remove old parameters, assign the names as buffers
#         for m, n in net._weights_module_names:
#             delattr(m, n)
#             m.register_buffer(n, None)

#         # register the flat one
#         net.register_parameter('flat_w', nn.Parameter(flat_w, requires_grad=True))

#         return net


# @add_metaclass(PatchModules)
# class ReparamModule(nn.Module):
#     def _apply(self, *args, **kwargs):
#         rv = super(ReparamModule, self)._apply(*args, **kwargs)
#         return rv

#     def get_param(self, clone=False):
#         if clone:
#             return self.flat_w.detach().clone().requires_grad_(self.flat_w.requires_grad)
#         return self.flat_w

#     @contextmanager
#     def unflatten_weight(self, flat_w):
#         ws = (t.view(s) for (t, s) in zip(flat_w.split(self._weights_numels), self._weights_shapes))
#         for (m, n), w in zip(self._weights_module_names, ws):
#             setattr(m, n, w)
#         yield
#         for m, n in self._weights_module_names:
#             setattr(m, n, None)

#     def forward_with_param(self, inp, new_w):
#         with self.unflatten_weight(new_w):
#             return nn.Module.__call__(self, inp)

#     def __call__(self, inp):
#         return self.forward_with_param(inp, self.flat_w)

#     # make load_state_dict work on both
#     # singleton dicts containing a flattened weight tensor and
#     # full dicts containing unflattened weight tensors...
#     def load_state_dict(self, state_dict, *args, **kwargs):
#         if len(state_dict) == 1 and 'flat_w' in state_dict:
#             return super(ReparamModule, self).load_state_dict(state_dict, *args, **kwargs)
#         with self.unflatten_weight(self.flat_w):
#             flat_w = self.flat_w
#             del self.flat_w
#             super(ReparamModule, self).load_state_dict(state_dict, *args, **kwargs)
#         self.register_parameter('flat_w', flat_w)

#     def reset(self, state, inplace=True):
#         if inplace:
#             flat_w = self.flat_w
#         else:
#             flat_w = torch.empty_like(self.flat_w).requires_grad_()
#         with torch.no_grad():
#             with self.unflatten_weight(flat_w):
#                 init_weights(self, state)
#         return flat_w

# class ImageClassifier(ReparamModule):
class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head, process_images=True):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        self.process_images = process_images
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def forward(self, inputs):
        if self.process_images:
            inputs = self.image_encoder(inputs)
        outputs = self.classification_head(inputs)
        return outputs

    def save(self, filename):
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)

