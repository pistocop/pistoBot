#!/usr/bin/env bash
#!/usr/bin/env bash
echo "--------------------"
echo "Download the public datasets"
echo "--------------------"

# TODO based on input download certain files
# TODO pdf2text

echo "Italian bible download..."
wget http://boldi.di.unimi.it/Corsi/ProgrMat2012/AppelloMag12/italiano.txt -O bible.txt
mv bible.txt ../


wget https://www.liberliber.it/mediateca/libri/b/bibbia/la_sacra_bibbia/pdf/bibbia_la_sacra_bibbia.pdf -o bibbia_la_sacra_bibbia.pdf
mv bibbia_la_sacra_bibbia.pdf ../
