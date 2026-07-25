from __future__ import annotations

from pathlib import Path
import re
from typing import Tuple

import numpy as np

from loquercio_px4_wrapper.model import LoquercioModelConfig, create_network, _tf


def resolve_checkpoint_prefix(path: str) -> str:
    raw = Path(path).expanduser()
    if raw.is_dir():
        tf = _tf()
        latest = tf.train.latest_checkpoint(str(raw))
        if latest:
            return latest
        raise FileNotFoundError(f"No TensorFlow checkpoint found in {raw}")
    if raw.suffix in ('.index', '.data'):
        return str(raw).split('.index')[0].split('.data')[0]
    if Path(str(raw) + '.index').exists():
        return str(raw)
    raise FileNotFoundError(
        f"Loquercio checkpoint prefix not found: {raw}. Expected {raw}.index or a checkpoint directory."
    )


def _checkpoint_value_key(prefix: str, weight) -> str:
    path = str(getattr(weight, "path", weight.name))
    leaf = path.rsplit("/", 1)[-1].split(":", 1)[0]
    return f"{prefix}/{leaf}/.ATTRIBUTES/VARIABLE_VALUE"


def _assign_checkpoint_layer(tf, checkpoint_prefix: str, layer, key_prefix: str) -> int:
    available = {
        name: tuple(shape)
        for name, shape in tf.train.list_variables(checkpoint_prefix)
        if name.startswith(key_prefix + "/") and ".OPTIMIZER_SLOT/" not in name
    }
    used = set()
    assigned = 0
    for weight in layer.weights:
        key = _checkpoint_value_key(key_prefix, weight)
        if key not in available:
            shape_matches = [
                name for name, shape in available.items()
                if name not in used and shape == tuple(weight.shape)
            ]
            if len(shape_matches) != 1:
                raise ValueError(
                    f"Missing unambiguous Loquercio tensor for {weight.path} "
                    f"under {key_prefix}; candidates={shape_matches}"
                )
            key = shape_matches[0]
        value = tf.train.load_variable(checkpoint_prefix, key)
        if tuple(value.shape) != tuple(weight.shape):
            raise ValueError(
                f"Loquercio checkpoint shape mismatch for {key}: "
                f"{tuple(value.shape)} != {tuple(weight.shape)}"
            )
        weight.assign(value)
        used.add(key)
        assigned += 1
    return assigned


def _restore_keras3_checkpoint(tf, model, checkpoint_prefix: str) -> int:
    """Load a TF/Keras 2 object checkpoint into the Keras 3 model."""
    variable_names = [
        name for name, _ in tf.train.list_variables(checkpoint_prefix)
        if ".OPTIMIZER_SLOT/" not in name
    ]
    backbone_indices = sorted({
        int(match.group(1))
        for name in variable_names
        if (match := re.match(r"net/backbone/0/layer_with_weights-(\d+)/", name))
    })
    backbone_layers = [layer for layer in model.backbone[0].layers if layer.weights]
    if len(backbone_indices) != len(backbone_layers):
        raise ValueError(
            "Loquercio backbone layer mismatch: checkpoint has "
            f"{len(backbone_indices)}, model has {len(backbone_layers)}"
        )

    assigned = 0
    for index, layer in zip(backbone_indices, backbone_layers):
        assigned += _assign_checkpoint_layer(
            tf, checkpoint_prefix, layer,
            f"net/backbone/0/layer_with_weights-{index}",
        )

    for group_name in (
        "resize_op", "img_mergenet", "resize_op_2",
        "states_conv", "resize_op_3", "plan_module",
    ):
        for index, layer in enumerate(getattr(model, group_name)):
            if layer.weights:
                assigned += _assign_checkpoint_layer(
                    tf, checkpoint_prefix, layer, f"net/{group_name}/{index}"
                )

    if assigned != len(model.weights):
        raise ValueError(
            f"Loaded {assigned} Loquercio weights, but model has {len(model.weights)}"
        )
    return assigned


class TensorFlowLoquercioBackend:
    def __init__(self, checkpoint_path: str, config: LoquercioModelConfig):
        self.tf = _tf()
        self.config = config
        self.network = create_network(config)
        self._warm_start_network()
        checkpoint_prefix = resolve_checkpoint_prefix(checkpoint_path)
        keras_major = int(str(self.tf.keras.__version__).split(".", 1)[0])
        if keras_major >= 3:
            self.loaded_weight_count = _restore_keras3_checkpoint(
                self.tf, self.network.model, checkpoint_prefix
            )
        else:
            checkpoint = self.tf.train.Checkpoint(net=self.network.model)
            status = checkpoint.restore(checkpoint_prefix)
            status.assert_existing_objects_matched()
            status.expect_partial()
            self.loaded_weight_count = len(self.network.model.weights)
        self.checkpoint_prefix = checkpoint_prefix

    def _warm_start_network(self) -> None:
        inputs = {
            'depth': np.zeros(
                (1, self.config.seq_len, self.config.img_height, self.config.img_width, 3),
                dtype=np.float32,
            ),
            'imu': np.zeros(
                (1, self.config.seq_len, self.config.raw_state_dim),
                dtype=np.float32,
            ),
        }
        _ = self.network(inputs)

    def infer(self, depth: np.ndarray, imu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inputs = {
            'depth': depth.astype(np.float32, copy=False),
            'imu': imu.astype(np.float32, copy=False),
        }
        pred = self.network(inputs).numpy()
        pred = pred[:, np.abs(pred[0, :, 0]).argsort(), :]
        alphas = np.abs(pred[0, :, 0])
        trajectories = pred[0, :, 1:]
        return alphas.astype(np.float32), trajectories.astype(np.float32)



class TfliteLoquercioBackend:
    def __init__(self, tflite_path: str, config: LoquercioModelConfig):
        self.config = config
        model_path = Path(tflite_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Loquercio TFLite model not found: {model_path}")
        self.interpreter = self._create_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.depth_input, self.imu_input = self._classify_inputs()
        self.output_index = self._select_output_index()
        self.model_path = str(model_path)

    def _create_interpreter(self, model_path: Path):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite import Interpreter
            except ImportError as exc:
                raise RuntimeError(
                    "TFLite runtime is required for backend='tflite'. Install "
                    "tflite_runtime or TensorFlow in the runtime environment."
                ) from exc
        return Interpreter(model_path=str(model_path))

    def _classify_inputs(self):
        depth_input = None
        imu_input = None
        for detail in self.input_details:
            name = str(detail.get('name', '')).lower()
            shape = tuple(int(x) for x in detail.get('shape', ()))
            if 'depth' in name or len(shape) == 5:
                depth_input = detail
            elif 'imu' in name or 'state' in name or len(shape) == 3:
                imu_input = detail
        if depth_input is None or imu_input is None:
            raise ValueError(
                "Could not identify TFLite depth and imu inputs from "
                f"{[(d.get('name'), d.get('shape')) for d in self.input_details]}"
            )
        return depth_input, imu_input

    def _select_output_index(self) -> int:
        for detail in self.output_details:
            shape = tuple(int(x) for x in detail.get('shape', ()))
            if len(shape) == 3:
                return int(detail['index'])
        if len(self.output_details) == 1:
            return int(self.output_details[0]['index'])
        raise ValueError(
            "Could not identify TFLite prediction output from "
            f"{[(d.get('name'), d.get('shape')) for d in self.output_details]}"
        )

    def infer(self, depth: np.ndarray, imu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.interpreter.set_tensor(int(self.depth_input['index']), depth.astype(np.float32, copy=False))
        self.interpreter.set_tensor(int(self.imu_input['index']), imu.astype(np.float32, copy=False))
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output_index).astype(np.float32, copy=False)
        pred = pred[:, np.abs(pred[0, :, 0]).argsort(), :]
        alphas = np.abs(pred[0, :, 0])
        trajectories = pred[0, :, 1:]
        return alphas.astype(np.float32), trajectories.astype(np.float32)
