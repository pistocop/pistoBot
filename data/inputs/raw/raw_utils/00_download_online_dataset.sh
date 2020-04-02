#!/usr/bin/env bash
#!/usr/bin/env bash
echo "--------------------"
echo "Download the public datasets"
echo "--------------------"

echo "Italian bible download..."
wget http://boldi.di.unimi.it/Corsi/ProgrMat2012/AppelloMag12/italiano.txt -O bible.txt
mv bible.txt ../
