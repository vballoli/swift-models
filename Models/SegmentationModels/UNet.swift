// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow

// Original paper:
// "U-Net: Convolutional Networks for Biomedical Image Segmentation"
// Olaf Ronneberger, Philipp Fischer, Thomas Brox
// https://arxiv.org/abs/1505.04597
// https://github.com/zhixuhao/unet/blob/master/model.py

public struct DoubleConv: Layer {
    public var conv1: Conv2D<Float>
    public var conv2: Conv2D<Float>

    public init(kernelSize: Int, inFilters: Int, outFilters: Int) {
        self.conv1 = Conv2D<Float>(filterShape: (kernelSize, kernelSize, inFilters, outFilters), activation: relu)
        self.conv2 = Conv2D<Float>(filterShape: (kernelSize, kernelSize, outFilters, outFilters), activation: relu)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: conv1, conv2)
    }
}

public struct DoubleConvPool: Layer {
    public var maxpool: MaxPool2D<Float>
    public var doubleConv: DoubleConv

    public init(kernelSize: Int, inFilters: Int, outFilters: Int) {
        self.maxpool = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
        self.doubleConv = DoubleConv(kernelSize: kernelSize, inFilters: inFilters, outFilters: outFilters)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: maxpool, doubleConv)
    }
}

public struct ConvUpSample: Layer {
    public var upsample: UpSampling2D<Float>
    public var conv: Conv2D<Float>

    public init(upsampleSize: Int, filterShape: (Int, Int, Int, Int)) {
        self.upsample = UpSampling2D<Float>(size: upsampleSize)
        self.conv = Conv2D<Float>(filterShape: filterShape, padding: .same, activation: relu)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: upsample, conv)
    }
}

public struct UNet: Layer {
    public var downDoubleConv1: DoubleConv
    public var downDoubleConvPool2: DoubleConvPool
    public var downDoubleConvPool3: DoubleConvPool
    public var downDoubleConvPool4: DoubleConvPool
    public var dropout4: Dropout<Float>
    //public var maxpool4: MaxPool2D<Float>
    public var downDoubleConvPool5: DoubleConvPool
    public var dropout5: Dropout<Float>
    public var convUpSample1: ConvUpSample
    public var upDoubleConv1: DoubleConv
    public var convUpSample2: ConvUpSample
    public var upDoubleConv2: DoubleConv
    public var convUpSample3: ConvUpSample
    public var upDoubleConv3: DoubleConv
    public var convUpSample4: ConvUpSample
    public var upDoubleConv4: DoubleConv
    public var conv9: Conv2D<Float>
    public var conv10: Conv2D<Float>

    public init() {
        self.downDoubleConv1 = DoubleConv(kernelSize: 3, inFilters: 1, outFilters: 64)
        self.downDoubleConvPool2 = DoubleConvPool(kernelSize: 3, inFilters: 64, outFilters: 128)
        self.downDoubleConvPool3 = DoubleConvPool(kernelSize: 3, inFilters: 128, outFilters: 256)
        self.downDoubleConvPool4 = DoubleConvPool(kernelSize: 3, inFilters: 256, outFilters: 512)
        self.dropout4 = Dropout<Float>(probability: 0.5)
        self.downDoubleConvPool5 = DoubleConvPool(kernelSize: 3, inFilters: 512, outFilters: 1024)
        self.dropout5 = Dropout<Float>(probability: 0.5)
        self.convUpSample1 = ConvUpSample(upsampleSize: 2, filterShape: (2, 2, 1024, 512))
        self.upDoubleConv1 = DoubleConv(kernelSize: 3, inFilters: 1024, outFilters: 512)
        self.convUpSample2 = ConvUpSample(upsampleSize: 2, filterShape: (2, 2, 512, 256))
        self.upDoubleConv2 = DoubleConv(kernelSize: 3, inFilters: 512, outFilters: 256)
        self.convUpSample3 = ConvUpSample(upsampleSize: 2, filterShape: (2, 2, 256, 128))
        self.upDoubleConv3 = DoubleConv(kernelSize: 3, inFilters: 256, outFilters: 128)
        self.convUpSample4 = ConvUpSample(upsampleSize: 2, filterShape: (2, 2, 128, 64))
        self.upDoubleConv4 = DoubleConv(kernelSize: 3, inFilters: 128, outFilters: 64)
        self.conv9 = Conv2D<Float>(filterShape: (2, 2, 64, 3), padding: .same, activation: relu)
        self.conv10 = Conv2D<Float>(filterShape: (1, 1, 3, 1))
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let conv1 = downDoubleConv1(input)
        let conv2 = downDoubleConvPool2(conv1)
        let conv3 = downDoubleConvPool3(conv2)
        let conv4 = conv3.sequenced(through: downDoubleConvPool4, dropout4)
        var up1 = conv4.sequenced(through: downDoubleConvPool5, dropout5, convUpSample1)
        up1 = up1.concatenated(with: conv4.slice(lowerBounds: [0, 4, 4, 0], upperBounds: [1, 60, 60, 512]), alongAxis: 3)
        var up2 = up1.sequenced(through: upDoubleConv1, convUpSample2)
        up2 = up2.concatenated(with: conv3.slice(lowerBounds: [0, 16, 16, 0], upperBounds: [1, 120, 120, 256]), alongAxis: 3)
        var up3 = up2.sequenced(through: upDoubleConv2, convUpSample3)
        up3 = up3.concatenated(with: conv2.slice(lowerBounds: [0, 40, 40, 0], upperBounds: [1, 240, 240, 128]), alongAxis: 3)
        var up4 = up3.sequenced(through: upDoubleConv3, convUpSample4)
        up4 = up4.concatenated(with: conv1.slice(lowerBounds: [0, 88, 88, 0], upperBounds: [1, 480, 480, 64]), alongAxis: 3)
        let convolved = up4.sequenced(through: upDoubleConv4, conv9, conv10)
        return convolved
    }
}