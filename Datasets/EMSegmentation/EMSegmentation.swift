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

//
//
//
//

import TensorFlow
import Foundation
import Python

#if canImport(FoundationNetworking)
    import FoundationNetworking
#endif

public struct EM {
    public let trainX: Tensor<UInt8>
    public let trainY: Tensor<UInt8>

    public init(
        localStorageDirectory: URL = FileManager.default.temporaryDirectory.appendingPathComponent(
            "EM", isDirectory: true)
    ) {
        self.trainY = fetchDataset(remoteURL: "http://brainiac2.mit.edu/isbi_challenge/sites/default/files/", localStorageDirectory: localStorageDirectory, imagesFileName: "train-volume", imagesFileExtension: "tif")
        self.trainX = fetchDataset(remoteURL: "http://brainiac2.mit.edu/isbi_challenge/sites/default/files/", localStorageDirectory: localStorageDirectory, imagesFileName: "test-volume", imagesFileExtension: "tif")
    }
}

func fetchDataset(
    remoteURL: String,
    localStorageDirectory: URL,
    imagesFileName: String,
    imagesFileExtension: String
) -> Tensor<UInt8> {
    guard let remoteRoot = URL(string: remoteURL) else {
        fatalError("Failed to create EM root url: \(remoteURL)")
    }

    let filePath = localStorageDirectory.appendingPathComponent(imagesFileName).appendingPathExtension(imagesFileExtension)
    // let destination = URL(fileURLWithPath: "/home/balli/")
    // let file = try! Data.init(contentsOf: remoteRoot)
    // print(file)
    // try! file.write(to: destination)
    let file = DatasetUtilities.downloadResource(
        filename: imagesFileName,
        fileExtension: imagesFileExtension,
        remoteRoot: remoteRoot,
        localStorageDirectory: localStorageDirectory,
        extract: false
    )

    let Image = Python.import("PIL.Image")
    let np = Python.import("numpy")

    func tensor(image: PythonObject, frame: Int) -> Tensor<UInt8>? {
        image.seek(frame)
        return Tensor(numpy: np.array(image))
    }

    func resizeImage(image: Tensor<Float>, toShape: [Int]) -> Tensor<Float> {
        return ZeroPadding2D(padding: (30, 30))(image.reshaped(to: TensorShape(toShape)))
    }

    let image = Tensor<Float>(tensor(image: Image.open(filePath.path), frame: 0) ?? Tensor<UInt8>([0]))
    var data = resizeImage(image: image, toShape: [1, 512, 512, 1])
    for i in 1..<30 {
        let im = Tensor<Float>(tensor(image: Image.open(filePath.path), frame: i) ?? Tensor<UInt8>([0]))
        let imData = resizeImage(image: im, toShape: [1, 512, 512, 1])
        data = data.concatenated(with: imData, alongAxis: 3)
    }
    return Tensor<UInt8>(data)
}