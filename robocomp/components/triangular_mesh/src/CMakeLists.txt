# Sources set
LIST(APPEND SOURCES
  ../src/specificworker.cpp
  ../src/navmap/core/NavMap.cpp
)

# Headers set
LIST(APPEND HEADERS
  ../src/specificworker.h
)

# Include directories for NavMap
INCLUDE_DIRECTORIES(../src)

# Find OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
MESSAGE(STATUS "OpenCV version: ${OpenCV_VERSION}")

LIST(APPEND LIBS ${LIBS} ${OpenCV_LIBS})

# no-char8_t to avoid a problem with ICE Connection lib.
add_definitions(-fmax-errors=1 -fno-char8_t)

