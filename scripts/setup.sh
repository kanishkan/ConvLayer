#!/bin/bash
# Check previous install
if [ -d ~/local/tce_llvm10 ]; then
	echo "TCE LLVM10 is already installed in the VM!"
	echo "Skipping installtion.. Contact TA if you have any specific issue."
	exit 1
fi

# Check sudo permission
if [[ $EUID -ne 0 ]]; then
	echo "Please run with sudo permission (i.e: sudo ./setup.sh)"
	exit 1
fi

# Install basic tools
sudo apt install gedit
python3 -m pip install --user imageio

# Update TCE Toolset
sudo apt update --allow-insecure-repositories
sudo apt-get install tcemc --allow-unauthenticated

# Get TCE-LLVM10
mkdir ~/local && cd ~/local
wget https://surfdrive.surf.nl/files/index.php/s/JSLzM8dkVcREskc/download

# Install TCE-LLVM10
mkdir tce_llvm10
dpkg -x ./download tce_llvm10

# Add the tools to system path
echo "export PATH=~/local/tce_llvm10/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=~/local/tce_llvm10/lib" >> ~/.bashrc

# Create temp folder for TCE Tools
mkdir ~/.tce

# Reload env
source ~/.bashrc
