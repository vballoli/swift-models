add_executable(UNet-EMSegmentation
  main.swift)
target_link_libraries(UNet-EMSegmentation PRIVATE
  SegmentationModels
  Datasets)


install(TARGETS UNet-EMSegmentation
  DESTINATION bin)
