import Datasets
import Foundation
import TensorFlow
import XCTest

final class EMSegmentationTests: XCTestCase {
    func testCreateEMDataset() {
        let dataset = EM()

        print(dataset.trainX)
        XCTAssertEqual(1, 2)
    }
}

extension EMSegmentationTests {
    static var allTests = [
        ("testCreateEMDataset", testCreateEMDataset)
    ]
}