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
import SegmentationModels
import Datasets

// Training UNet model
// 1. Load data
// 2. Define model
// 3. Train
// 4. Test
// 5. Publish   

var model = UNet()
var dataset = EM()

var lr: Float = 1e-3
var optimizer = Adam(for: model, learningRate: lr)

func accuracy(predictions: Tensor<UInt8>, truths: Tensor<UInt8>) -> Float {
    return Tensor<Float>(predictions .== truths).mean().scalarized()
}

let epochCount = 500
var trainAccuracyResults: [Float] = []
var trainLossResults: [Float] = []

for epoch in 1...epochCount {
    var epochLoss: Float = 0
    var epochAccuracy: Float = 0
    var batchCount: Int = 0

    for i in 0..<30 {
        print("aksdfjadjfs")
        let grad = gradient(at: model) { model -> Tensor<Float> in
            print("aksdfjadjfs")
            let logits = model(dataset.trainX[i].reshaped(to: [1, 572, 572, 1]))
            print("aksdfjadjfs")
            let loss = sigmoidCrossEntropy(logits: logits.reshaped(to: [388*388, 1]), labels: dataset.trainY[i].reshaped(to: [388*388]))
            print("aksdfjadjfs")
            //epochLoss += loss.scalarized()
            print("aksdfjadjfs")
            //epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: dataset.trainY)
            return loss
        }
        optimizer.update(&model, along: grad)
    }
    // let logits = model(batch.features)
    // epochLoss += loss.scalarized()
    // batchCount += 1
    // epochAccuracy /= Float(batchCount)
    // epochLoss /= Float(batchCount)
    // trainAccuracyResults.append(epochAccuracy)
    // trainLossResults.append(epochLoss)
    // if epoch % 50 == 0 {
    //     print("Epoch \(epoch): Loss: \(epochLoss), Accuracy: \(epochAccuracy)")
    // }
}