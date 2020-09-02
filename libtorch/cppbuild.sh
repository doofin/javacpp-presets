#!/bin/bash
# This file is meant to be included by the parent cppbuild.sh script
if [[ -z "$PLATFORM" ]]; then
    pushd ..
    bash cppbuild.sh "$@" libtorch
    popd
    exit
fi

#if [[ $PLATFORM == XXX* ]]; then
#    #No XXX support yet
#    echo "Error: Platform \"$PLATFORM\" is not supported"
#    exit 1
#fi

export LIBTORCH_VERSION=1.6.0

download https://download.pytorch.org/libtorch/cpu/libtorch-macos-$LIBTORCH_VERSION.zip libtorch-macos-$LIBTORCH_VERSION.zip

# TODO: Linux/Windows builds with GPU (CUDA 10.2 for 1.6.0) support.
#download https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$LIBTORCH_VERSION.zip
#download https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-$LIBTORCH_VERSION.zip

mkdir -p "$PLATFORM"
cd "$PLATFORM"
INSTALL_PATH=`pwd`

echo "Decompressing archives..."
unzip ../libtorch-macos-$LIBTORCH_VERSION.zip

#export LIBRARY_PATH="$INSTALL_PATH/lib"
#export PATH="$INSTALL_PATH/bin:$PATH"
#export CFLAGS="-I$INSTALL_PATH/include"
#export CXXFLAGS="-I$INSTALL_PATH/include"

cd ../..
