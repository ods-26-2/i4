cmake_minimum_required(VERSION 3.16)
project(server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Crow REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET)

add_executable(server src/server.cpp)
target_include_directories(server PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(server PRIVATE
    ${OpenCV_LIBS}
    Crow::Crow
    PkgConfig::FFMPEG
    pthread
    rt)