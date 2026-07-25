from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoquercioModelConfig:
    img_width: int = 224
    img_height: int = 224
    seq_len: int = 1
    modes: int = 3
    state_dim: int = 3
    out_seq_len: int = 10
    use_rgb: bool = False
    use_depth: bool = True
    use_position: bool = False
    use_attitude: bool = True
    use_bodyrates: bool = True
    freeze_backbone: bool = False

    @property
    def raw_state_dim(self) -> int:
        return 21 if self.use_bodyrates else 18

    @property
    def output_dim_per_mode(self) -> int:
        return 1 + self.state_dim * self.out_seq_len


class TensorFlowUnavailable(RuntimeError):
    pass


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise TensorFlowUnavailable(
            "TensorFlow is required for backend='tensorflow'. Install TensorFlow "
            "in the AirStack robot container or use a converted runtime backend."
        ) from exc
    return tf


def create_network(config: LoquercioModelConfig):
    tf = _tf()
    return PlaNet(tf, config)


class PlaNet:
    """Keras model structure used by uzh-rpg/agile_autonomy planner_learning."""

    def __init__(self, tf, config: LoquercioModelConfig):
        self.tf = tf
        self.config = config
        self.model = self._create_model()

    def __call__(self, inputs):
        return self.model(inputs)

    @property
    def trainable_variables(self):
        return self.model.trainable_variables

    @property
    def variables(self):
        return self.model.variables

    def _create_model(self):
        tf = self.tf
        config = self.config

        class _PlaNetModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                channels = 3 * int(config.use_rgb) + 3 * int(config.use_depth)
                input_size = (config.img_height, config.img_width, channels)

                if config.use_rgb or config.use_depth:
                    self.backbone = [
                        tf.keras.applications.MobileNet(
                            include_top=False,
                            weights=None,
                            input_shape=input_size,
                            pooling=None,
                        )
                    ]
                    self.backbone[0].trainable = not config.freeze_backbone
                    self.resize_op = [tf.keras.layers.Conv1D(128, 1, padding='valid')]
                    self.img_mergenet = [
                        tf.keras.layers.Conv1D(128, 2, padding='same'),
                        tf.keras.layers.LeakyReLU(alpha=1e-2),
                        tf.keras.layers.Conv1D(64, 2, padding='same'),
                        tf.keras.layers.LeakyReLU(alpha=1e-2),
                        tf.keras.layers.Conv1D(64, 2, padding='same'),
                        tf.keras.layers.LeakyReLU(alpha=1e-2),
                        tf.keras.layers.Conv1D(32, 2, padding='same'),
                        tf.keras.layers.LeakyReLU(alpha=1e-2),
                    ]
                    self.resize_op_2 = [
                        tf.keras.layers.Conv1D(config.modes, 3, padding='valid')
                    ]

                self.states_conv = [
                    tf.keras.layers.Conv1D(64, 2, padding='same'),
                    tf.keras.layers.LeakyReLU(alpha=.5),
                    tf.keras.layers.Conv1D(32, 2, padding='same'),
                    tf.keras.layers.LeakyReLU(alpha=.5),
                    tf.keras.layers.Conv1D(32, 2, padding='same'),
                    tf.keras.layers.LeakyReLU(alpha=.5),
                    tf.keras.layers.Conv1D(32, 2, padding='same'),
                ]
                self.resize_op_3 = [
                    tf.keras.layers.Conv1D(config.modes, 3, padding='valid')
                ]
                output_dim = config.output_dim_per_mode
                self.plan_module = [
                    tf.keras.layers.Conv1D(64, 1, padding='valid'),
                    tf.keras.layers.LeakyReLU(alpha=.5),
                    tf.keras.layers.Conv1D(128, 1, padding='valid'),
                    tf.keras.layers.LeakyReLU(alpha=.5),
                    tf.keras.layers.Conv1D(128, 1, padding='valid'),
                    tf.keras.layers.LeakyReLU(alpha=.5),
                    tf.keras.layers.Conv1D(output_dim, 1, padding='same'),
                ]

            def call(self, inputs):
                if config.use_position:
                    imu_obs = inputs['imu']
                else:
                    imu_obs = inputs['imu'][:, :, 3:]
                if not config.use_attitude:
                    if config.use_position:
                        raise ValueError('Loquercio config cannot use position without attitude.')
                    imu_obs = inputs['imu'][:, :, 12:]

                imu_embeddings = self._imu_branch(imu_obs)
                img_embeddings = self._preprocess_frames(inputs)
                if img_embeddings is not None:
                    total_embeddings = tf.concat((img_embeddings, imu_embeddings), axis=-1)
                else:
                    total_embeddings = imu_embeddings
                return self._plan_branch(total_embeddings)

            def _conv_branch(self, image):
                x = tf.keras.applications.mobilenet.preprocess_input(image)
                for layer in self.backbone:
                    x = layer(x)
                x = tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1]))
                for layer in self.resize_op:
                    x = layer(x)
                return tf.reshape(x, (tf.shape(x)[0], -1))

            def _image_branch(self, img_seq):
                img_fts = tf.map_fn(
                    self._conv_branch,
                    elems=img_seq,
                    parallel_iterations=config.seq_len,
                    fn_output_signature=tf.float32,
                )
                img_fts = tf.transpose(img_fts, (1, 0, 2))
                x = img_fts
                for layer in self.img_mergenet:
                    x = layer(x)
                x = tf.transpose(x, (0, 2, 1))
                for layer in self.resize_op_2:
                    x = layer(x)
                return tf.transpose(x, (0, 2, 1))

            def _imu_branch(self, embeddings):
                x = embeddings
                for layer in self.states_conv:
                    x = layer(x)
                x = tf.transpose(x, (0, 2, 1))
                for layer in self.resize_op_3:
                    x = layer(x)
                return tf.transpose(x, (0, 2, 1))

            def _plan_branch(self, embeddings):
                x = embeddings
                for layer in self.plan_module:
                    x = layer(x)
                return x

            def _preprocess_frames(self, inputs):
                if config.use_rgb and config.use_depth:
                    img_seq = tf.concat((inputs['rgb'], inputs['depth']), axis=-1)
                elif config.use_rgb:
                    img_seq = inputs['rgb']
                elif config.use_depth:
                    img_seq = inputs['depth']
                else:
                    return None
                img_seq = tf.transpose(img_seq, (1, 0, 2, 3, 4))
                return self._image_branch(img_seq)

        return _PlaNetModel()
