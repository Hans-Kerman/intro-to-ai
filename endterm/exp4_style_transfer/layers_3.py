import numpy as np

class StyleLossLayer(object):
    def __init__(self):
        print('\tStyle loss layer.')
        self.G_input = None
        self.G_style = None
        self.input_shape = None
        self.style_shape = None
    
    def gram_matrix(self, x):
        N, C, H, W = x.shape
        features = x.reshape(N, C, -1)  # 转换为(N, C, H*W)
        return np.matmul(features, features.transpose(0,2,1)) / (C * H * W)
    
    def forward(self, input_layer, style_layer):
        self.input_shape = input_layer.shape
        self.style_shape = style_layer.shape
        self.G_input = self.gram_matrix(input_layer)
        self.G_style = self.gram_matrix(style_layer)
        loss = np.sum((self.G_input - self.G_style)**2) / 4
        return loss
    
    def backward(self, input_layer, style_layer):
        if self.input_shape is None:
            raise RuntimeError("Must call forward before backward")
        N, C, H, W = self.input_shape
        M = H * W
        G_diff = self.G_input - self.G_style
        x_flat = input_layer.reshape(N, C, -1)  # (N, C, M)
        dX_flat = np.zeros_like(x_flat)
        for i in range(N):
            dX_flat[i] = np.dot(G_diff[i], x_flat[i]) / (C * M)
        return dX_flat.reshape(self.input_shape)

class ContentLossLayer(object):
    def __init__(self):
        print('\tContent loss layer.')
    
    def forward(self, input_layer, content_layer):
        self.input = input_layer
        self.content = content_layer
        loss = 0.5 * np.sum((input_layer - content_layer) ** 2)
        return loss
    
    def backward(self, input_layer, content_layer):
        bottom_diff = input_layer - content_layer
        return bottom_diff