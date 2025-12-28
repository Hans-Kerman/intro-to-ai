import numpy as np
import time

def show_matrix(mat, name):
    pass

def show_time(time, name):
    pass

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, type=0):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.forward = self.forward_speedup
        self.backward = self.backward_speedup
        print(f'\tConvolutional layer with kernel size {kernel_size}, input channel {channel_in}, output channel {channel_out}.')

    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, 
                                     size=(self.channel_out, self.channel_in, self.kernel_size, self.kernel_size))
        self.bias = np.zeros([self.channel_out])

    def im2col(self, input_data):
        N, C, H, W = input_data.shape
        out_h = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        
        img = np.pad(input_data, [(0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)], 
                    mode='constant')
        col = np.zeros((N, C, self.kernel_size, self.kernel_size, out_h, out_w))
        
        for y in range(self.kernel_size):
            y_max = y + self.stride*out_h
            for x in range(self.kernel_size):
                x_max = x + self.stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:self.stride, x:x_max:self.stride]
        
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col, (N, out_h, out_w)

    def forward_speedup(self, input_data):
        start_time = time.time()
        self.input_shape = input_data.shape
        N, C, H, W = input_data.shape
        
        # 计算输出尺寸
        out_h = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # 使用im2col转换
        self.col, _ = self.im2col(input_data)
        self.weight_col = self.weight.reshape(self.channel_out, -1).T
        
        # 矩阵乘法计算卷积
        output = np.dot(self.col, self.weight_col) + self.bias
        output = output.reshape(N, out_h, out_w, self.channel_out).transpose(0, 3, 1, 2)
        
        self.forward_time = time.time() - start_time
        return output

    def backward_speedup(self, top_diff):
        start_time = time.time()
        N, C, H, W = self.input_shape
        
        # 重塑梯度
        top_diff_reshaped = top_diff.transpose(0, 2, 3, 1).reshape(-1, self.channel_out)
        
        # 计算权重梯度
        d_weight = np.dot(self.col.T, top_diff_reshaped)
        d_weight = d_weight.T.reshape(self.weight.shape)
        
        # 计算偏置梯度
        d_bias = np.sum(top_diff_reshaped, axis=0)
        
        # 计算输入梯度
        d_col = np.dot(top_diff_reshaped, self.weight_col.T)
        d_input = self.col2im(d_col, self.input_shape)
        
        self.backward_time = time.time() - start_time
        return d_input

    def col2im(self, col, input_shape):
        N, C, H, W = input_shape
        out_h = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        
        col_reshaped = col.reshape(N, out_h, out_w, C, self.kernel_size, self.kernel_size)
        col_reshaped = col_reshaped.transpose(0, 3, 4, 5, 1, 2)
        
        img = np.zeros((N, C, H + 2*self.padding + self.stride - 1, W + 2*self.padding + self.stride - 1))
        
        for y in range(self.kernel_size):
            for x in range(self.kernel_size):
                img[:, :, y:y + self.stride*out_h:self.stride, x:x + self.stride*out_w:self.stride] += col_reshaped[:, :, y, x, :, :]
        
        return img[:, :, self.padding:H + self.padding, self.padding:W + self.padding]

    def load_param(self, weight, bias):
        # 确保权重形状匹配 (C_in, H, W, C_out) -> (C_out, C_in, H, W)
        weight = np.transpose(weight, [3, 0, 1, 2])
        self.weight = weight
        self.bias = bias

    def get_forward_time(self):
        return self.forward_time
        
    def get_backward_time(self):
        return self.backward_time

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, type=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.forward = self.forward_speedup
        self.backward = self.backward_speedup
        print(f'\tMax pooling layer with kernel size {kernel_size}, stride {stride}.')

    def forward_speedup(self, input_data):
        start_time = time.time()
        self.input = input_data
        N, C, H, W = input_data.shape
        
        # 计算输出尺寸
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1
        
        # 创建视图进行池化
        output = np.zeros((N, C, out_h, out_w))
        self.max_indices = np.zeros((N, C, out_h, out_w, 2), dtype=int)
        
        for h in range(out_h):
            for w in range(out_w):
                h_start = h * self.stride
                h_end = h_start + self.kernel_size
                w_start = w * self.stride
                w_end = w_start + self.kernel_size
                
                window = input_data[:, :, h_start:h_end, w_start:w_end]
                window_reshaped = window.reshape(N, C, -1)
                
                # 找到最大值和索引
                output[:, :, h, w] = np.max(window_reshaped, axis=2)
                max_indices_flat = np.argmax(window_reshaped, axis=2)
                
                # 转换为2D索引
                for n in range(N):
                    for c in range(C):
                        idx = max_indices_flat[n, c]
                        h_idx, w_idx = np.unravel_index(idx, (self.kernel_size, self.kernel_size))
                        self.max_indices[n, c, h, w] = [h_start + h_idx, w_start + w_idx]
        
        self.forward_time = time.time() - start_time
        return output

    def backward_speedup(self, top_diff):
        start_time = time.time()
        bottom_diff = np.zeros_like(self.input)
        
        # 使用预存的最大值索引
        for n in range(top_diff.shape[0]):
            for c in range(top_diff.shape[1]):
                for h in range(top_diff.shape[2]):
                    for w in range(top_diff.shape[3]):
                        h_idx, w_idx = self.max_indices[n, c, h, w]
                        bottom_diff[n, c, h_idx, w_idx] += top_diff[n, c, h, w]
        
        self.backward_time = time.time() - start_time
        return bottom_diff

    def get_forward_time(self):
        return self.forward_time
        
    def get_backward_time(self):
        return self.backward_time

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print(f'\tFlatten layer with input shape {input_shape}, output shape {output_shape}.')

    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        return self.output

    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        return bottom_diff