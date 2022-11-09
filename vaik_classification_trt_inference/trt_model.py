from typing import List, Dict, Tuple, Union

from PIL import Image
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TrtModel:
    def __init__(self, input_saved_model_path: str = None, classes: Tuple = None, preprocess: str = None):
        self.preprocess = preprocess
        self.classes = classes
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(input_saved_model_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                "index": i,
                "name": name,
                "dtype": np.dtype(trt.nptype(dtype)),
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def inference(self, input_image_list: List[np.ndarray]) -> Tuple[List[Dict], np.ndarray]:
        resized_image_array = self.__preprocess_image_list(input_image_list,
                                                           (self.inputs[0]['shape'][1], self.inputs[0]['shape'][2]))
        raw_pred = self.__inference(resized_image_array)
        output = self.__output_parse(raw_pred)
        return output, raw_pred

    def __inference(self, resize_input_tensor: np.ndarray) -> np.ndarray:
        if len(resize_input_tensor.shape) != 4:
            raise ValueError('dimension mismatch')
        if self.preprocess == 'V2':
            resize_input_tensor = (resize_input_tensor / 128.0) / 128.0

        model_input_dtype = self.inputs[0]['dtype']
        output_spec = self.__output_spec()
        output_tensor = np.zeros((resize_input_tensor.shape[0], *output_spec[0][1:]), output_spec[1])
        for index in range(0, resize_input_tensor.shape[0], self.inputs[0]['shape'][0]):
            batch = resize_input_tensor[index:index + self.inputs[0]['shape'][0], :, :, :]
            batch_pad = np.zeros(self.inputs[0]['shape'], batch.dtype)
            batch_pad[:batch.shape[0], :, :, :] = batch.astype(model_input_dtype)
            output_tensor[index:index + batch.shape[0]] = self.__inference_tensor(batch_pad)[:batch.shape[0]]
        return output_tensor

    def __inference_tensor(self, input_array: np.ndarray):
        # Prepare the output data
        output = np.zeros(*self.__output_spec())

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]["allocation"], np.ascontiguousarray(input_array.astype(self.inputs[0]['dtype'])))
        self.context.execute_v2(self.allocations)
        cuda.memcpy_dtoh(output, self.outputs[0]["allocation"])
        return output

    def __output_spec(self):
        return self.outputs[0]["shape"], self.outputs[0]["dtype"]

    def __preprocess_image_list(self, input_image_list: List[np.ndarray],
                                resize_input_shape: Tuple[int, int]) -> np.ndarray:
        resized_image_list = []
        resized_scales_list = []
        for input_image in input_image_list:
            resized_image, resized_scales = self.__preprocess_image(input_image, resize_input_shape)
            resized_image_list.append(resized_image)
            resized_scales_list.append(resized_scales)
        return np.stack(resized_image_list)

    def __preprocess_image(self, input_image: np.ndarray, resize_input_shape: Tuple[int, int]) -> Tuple[
        np.ndarray, Tuple[float, float]]:
        if len(input_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_image.dtype}')

        output_image = np.zeros((*resize_input_shape, input_image.shape[2]),
                                dtype=input_image.dtype)
        resized_scale = min(resize_input_shape[1] / input_image.shape[1],
                            resize_input_shape[0] / input_image.shape[0])
        pil_image = Image.fromarray(input_image)
        x_ratio, y_ratio = resize_input_shape[1] / pil_image.width, resize_input_shape[0] / pil_image.height
        if x_ratio < y_ratio:
            resize_size = (resize_input_shape[1], round(pil_image.height * x_ratio))
        else:
            resize_size = (round(pil_image.width * y_ratio), resize_input_shape[0])
        resize_pil_image = pil_image.resize(resize_size)
        resize_image = np.array(resize_pil_image)
        output_image[:resize_image.shape[0], :resize_image.shape[1], :] = resize_image
        return output_image, (resize_input_shape[1] / resized_scale, resize_input_shape[0] / resized_scale)

    def __output_parse(self, pred: np.ndarray) -> List[Dict]:
        output_dict_list = []
        pred_index = np.argsort(-pred, axis=-1)
        for index in range(pred.shape[0]):
            output_dict = {'score': pred[index][pred_index[index]].tolist(),
                           'label': [self.classes[class_index] for class_index in pred_index[index]]}
            output_dict_list.append(output_dict)
        return output_dict_list