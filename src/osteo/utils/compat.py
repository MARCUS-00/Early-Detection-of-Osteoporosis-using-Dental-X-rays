import tensorflow as tf


def load_model_compat(model_path: str):
    class _CompatDense(tf.keras.layers.Dense):
        @classmethod
        def from_config(cls, config):
            config.pop('quantization_config', None)
            return super().from_config(config)

    class _CompatBatchNorm(tf.keras.layers.BatchNormalization):
        @classmethod
        def from_config(cls, config):
            config.pop('quantization_config', None)
            return super().from_config(config)

    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            'Dense': _CompatDense,
            'BatchNormalization': _CompatBatchNorm,
        },
    )
