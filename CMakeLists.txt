cmake_minimum_required(VERSION 2.8.4)
project(ga)

# COMPILER: COMMON FLAGS
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -m64 -Wall -Werror")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG=true -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -DDEBUG=false -O3 -flto")

# COMPILER: PLATFORM-DEPENDENT FLAGS
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -save-temps=obj")
endif()

# LOAD BOOST
find_package(Boost)

set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")

# INCLUDE DIRECTORIES
include_directories("src")
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

# BUILD THE EXECUTABLE INSIDE ./build
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

set(SOURCE_FILES
    src/ga.hpp
)

add_executable(ga ${SOURCE_FILES})