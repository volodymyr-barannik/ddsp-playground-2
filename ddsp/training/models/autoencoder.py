# Copyright 2022 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model that encodes audio features and decodes with a ddsp processor group."""

import ddsp
from ddsp.training.models.model import Model


class Autoencoder(Model):
  """Wrap the model function for dependency injection with gin."""

    def __init__(self,
                 preprocessor=None,     # f0, l
                 encoder=None,          # z (optional)
                 decoder=None,          # (f0, l, z) --> synth_params
                 processor_group=None,  # synth_params --> audio
                 losses=None,           # [spectral_loss]
                 **kwargs):
        super().__init__(**kwargs)
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.processor_group = processor_group
        self.loss_objs = ddsp.core.make_iterable(losses)

        print("created autoencoder")

    def encode(self, features, training=True):
        """Get conditioning by preprocessing then encoding."""
        if self.preprocessor is not None:
            print(f"debug: encode: before preprocessor: features['audio'].shape={features['audio'].shape}")
            features.update(self.preprocessor(features, training=training))
            print(f"debug: encode: after preprocessor: features['audio'].shape={features['audio'].shape}")
        if self.encoder is not None:
            features.update(self.encoder(features))
        return features

    def decode(self, features, training=True):
        """Get generated audio by decoding than processing."""
        print(f"debug: encode: before decoder: features['audio'].shape={features['audio'].shape}")
        features.update(self.decoder(features, training=training))
        print(f"debug: encode: after decoder: features['audio'].shape={features['audio'].shape}")
        return self.processor_group(features)

    def get_audio_from_outputs(self, outputs):
        """Extract audio output tensor from outputs dict of call()."""
        return outputs['audio_synth']

    def call(self, features, training=True):
        """Run the core of the network, get predictions and loss."""
        features = self.encode(features, training=training)
        features.update(self.decoder(features, training=training))

        # Run through processor group.
        pg_out = self.processor_group(features, return_outputs_dict=True)

        # Parse outputs
        outputs = pg_out['controls']
        outputs['audio_synth'] = pg_out['signal']

        if training:
            self._update_losses_dict(
                self.loss_objs, features['audio'], outputs['audio_synth'])

        return outputs
