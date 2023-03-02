import numpy as np
from IPython.display import Audio
from IPython.core.display import display

import ddsp


# Alias these for backwards compatibility and ease.
specplot = ddsp.training.plotting.specplot
plot_impulse_responses = ddsp.training.plotting.plot_impulse_responses
transfer_function = ddsp.training.plotting.transfer_function

DEFAULT_SAMPLE_RATE = ddsp.spectral_ops.CREPE_SAMPLE_RATE


def play(array_of_floats,
         sample_rate=DEFAULT_SAMPLE_RATE,
         ephemeral=True,
         autoplay=False):
    """Creates an HTML5 audio widget to play a sound in Colab.

  This function should only be called from a Colab notebook.

  Args:
    array_of_floats: A 1D or 2D array-like container of float sound samples.
      Values outside of the range [-1, 1] will be clipped.
    sample_rate: Sample rate in samples per second.
    ephemeral: If set to True, the widget will be ephemeral, and disappear on
      reload (and it won't be counted against realtime document size).
    autoplay: If True, automatically start playing the sound when the widget is
      rendered.
  """
    # If batched, take first element.
    if len(array_of_floats.shape) == 2:
        array_of_floats = array_of_floats[0]

    normalizer = float(np.iinfo(np.int16).max)
    array_of_ints = np.array(
        np.asarray(array_of_floats) * normalizer, dtype=np.int16)

    display(Audio(data=array_of_ints, rate=sample_rate, autoplay=autoplay))
