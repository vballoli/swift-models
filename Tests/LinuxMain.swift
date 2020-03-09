import XCTest

import ImageClassificationTests
import SegmentationModelsTests
import MiniGoTests
import FastStyleTransferTests
import DatasetsTests
import CheckpointTests

var tests = [XCTestCaseEntry]()
tests += SegmentationModelsTests.allTests()
tests += DatasetsTests.allTests()
tests += ImageClassificationTests.allTests()
tests += MiniGoTests.allTests()
tests += FastStyleTransferTests.allTests()
tests += CheckpointTests.allTests()
XCTMain(tests)
