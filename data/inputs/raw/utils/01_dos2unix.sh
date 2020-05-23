#!/usr/bin/env bash
cd ..

echo "--------------------"
echo "Normalize the files under /data/inputs/raw/ directory"
echo "This include: [CRLF to LF]"
echo "--------------------"

echo "[Starting file format]"
echo ""
file *

echo "---"
echo "Installing required packages..."
packages='dos2unix'
if ! dpkg -s ${packages} >/dev/null 2>&1; then
  sudo apt-get install ${packages}
fi

echo "Converting CRLF to LF..."
dos2unix *.txt >/dev/null 2>&1


echo "---"
echo "[Check files format]"
echo ""
file *