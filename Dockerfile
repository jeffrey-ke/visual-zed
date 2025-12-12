from stereolabs/zed:3.7-gl-devel-cuda11.4-ubuntu20.04

arg DEBIAN_FRONTEND=noninteractive

run sudo apt update && sudo apt install -y usbutils
