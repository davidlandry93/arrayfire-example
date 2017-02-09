
#include <iostream>

#include <arrayfire.h>

int main(int argc, char *argv[])
{
    std::cout << "Arrayfire example" << std::endl;

    // Print what backend arrayfire is plugged into.
    af::info();

    // Create a point cloud.
    int pointcloud_size = 10000;
    af::array points = af::randu(pointcloud_size, 3, f32);

    af::array V, D, E;
    auto mean = af::mean(points, 0);

    af::print("Mean", mean);

    auto scaled = points / 5.0;
    mean = af::mean(scaled, 0);

    af::print("Scaled mean", mean);

    return 0;
}

