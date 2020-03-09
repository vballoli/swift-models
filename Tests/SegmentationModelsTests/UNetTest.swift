import TensorFlow
import XCTest

@testable import SegmentationModels

final class UNetTest: XCTestCase {
    override class func setUp() {
        Context.local.learningPhase = .inference
    }

    func testUNet() {
        let input = Tensor<Float>(
            randomNormal: [1, 572, 572, 1], mean: Tensor<Float>(0.5),
            standardDeviation: Tensor<Float>(0.1), seed: (0xffeffe, 0xfffe))
        let unet = UNet()
        let unetResult = unet(input)
        print(unetResult.shape)
        XCTAssertEqual(unetResult.shape, [1, 388, 388, 2])
    }
}

extension UNetTest {
    static var allTests = [
        ("testUNet", testUNet),
    ]
}