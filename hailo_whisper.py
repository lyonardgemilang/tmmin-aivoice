# hailo_whisper_encoder.py
import numpy as np
from hailo_platform import HEF, VDevice, InferVStreams, VStreamsParams, FormatType

class HailoWhisperEncoder:
    def __init__(self, hef_path: str, float_output: bool = True):
        # Open device and configure the HEF
        self.vdev = VDevice(); self.vdev.__enter__()
        self.hef = HEF(hef_path)
        self.net_groups = self.vdev.configure(self.hef, self.hef.create_configure_params())
        self.net_group = self.net_groups[0]

        in_infos  = self.hef.get_input_vstream_infos()
        out_infos = self.hef.get_output_vstream_infos()
        self.in_name  = in_infos[0].name
        self.out_name = out_infos[0].name

        params = VStreamsParams.make_from_hef(
            self.hef,
            quantized=not float_output,
            output_format_type=FormatType.FLOAT32 if float_output else FormatType.UINT8
        )
        self.runner = InferVStreams(self.net_group, in_infos, out_infos, params)
        self.runner.__enter__()

    def encode(self, input_features: np.ndarray) -> np.ndarray:
        # input_features shape must be (1, 80, T), float32
        result = self.runner.infer({self.in_name: input_features})
        return result[self.out_name]  # expected (1, T_enc, d_model)

    def close(self):
        try:
            self.runner.__exit__(None, None, None)
        finally:
            self.vdev.__exit__(None, None, None)
