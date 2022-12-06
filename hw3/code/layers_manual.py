import math

import layers_keras
import numpy as np
import tensorflow as tf

BatchNormalization = layers_keras.BatchNormalization
Dropout = layers_keras.Dropout


class Conv2D(layers_keras.Conv2D):
    """
    Manually applies filters using the appropriate filter size and stride size
    """

    def call(self, inputs, training=False):
        # If it's training, revert to layers implementation since this can be non-differentiable
        if training:
            return super().call(inputs, training)

        # Otherwise, manually compute convolution at inference.
        # Doesn't have to be differentiable. YAY!
        # Batch #, height, width, # channels in input
        bn, h_in, w_in, c_in = inputs.shape
        c_out = self.filters  # channels in output
        fh, fw = self.kernel_size  # filter height & width
        sh, sw = self.strides  # filter stride

        # Cleaning padding input.
        if self.padding == "SAME":
            out_h = h_in // sh
            out_w = w_in // sw

            if (h_in % sh == 0):
                pad_along_height = max(fh - sh, 0)
            else:
                pad_along_height = max(fh - (h_in % sh), 0)

            if (w_in % sw == 0):
                pad_along_width = max(fw - sw, 0)
            else:
                pad_along_width = max(fw - (w_in % sw), 0)

            pt = pad_along_height // 2
            pb = pad_along_height - pt
            pl = pad_along_width // 2
            pr = pad_along_width - pl
        elif self.padding == "VALID":
            out_h = (h_in - fh) // sh + 1
            out_w = (w_in - fw) // sw + 1

            pt, pb, pl, pr = 0, 0, 0, 0
        else:
            raise AssertionError(f"Illegal padding type {self.padding}")

        # TODO: Convolve filter from above with the inputs.
        # Note: Depending on whether you used SAME or VALID padding,
        # the input and output sizes may not be the same

        # Pad input if necessary
        padded_input = tf.pad(inputs,
                              tf.constant(
                                  [[0, 0], [pt, pb], [pl, pr], [0, 0]]), 'constant')

        # Iterate and apply convolution operator to each image
        output = np.zeros((bn, out_h, out_w, c_out))

        for b in range(0, bn):
            for h in range(0, out_h):
                for w in range(0, out_w):
                    for c in range(0, c_out):
                        output[b, h, w, c] = np.sum(
                            padded_input[b, (h * sh):((h * sh) + fh), (w * sw):((w * sw) + fw), :] * self.kernel[:, :, :, c])

        return tf.convert_to_tensor(output, tf.float32)
