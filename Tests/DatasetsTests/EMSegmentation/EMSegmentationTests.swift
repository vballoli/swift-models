import Datasets
import Foundation
import TensorFlow
import XCTest

final class EMSegmentationTests: XCTestCase {
    func testCreateEMDataset() {
        let dataset = EM()

        XCTAssertEqual(dataset.trainX.shape, [1, 572, 572, 30]);
        XCTAssertEqual(dataset.trainY.shape, [1, 388, 388, 30]); 
    }
}

extension EMSegmentationTests {
    static var allTests = [
        ("testCreateEMDataset", testCreateEMDataset)
    ]
}