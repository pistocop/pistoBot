#!/usr/bin/env bash
#!/usr/bin/env bash
cd ..

echo "--------------------"
echo "Convert file input from latin to utf-8 encoding"
echo "--------------------"

latinFiles=( "$@" )

echo "[Starting file format]"
echo ""
file *

echo "---"
echo "Installing required packages..."
packages='iconv'
if ! dpkg -s ${packages} >/dev/null 2>&1; then
  sudo apt-get install ${packages} >/dev/null 2>&1
fi

for file in ${latinFiles[@]} ; do
    if test -f "$file"; then
        echo "Converting '$file' to utf-8..."
        iconv -f iso-8859-1 -t utf8 -o "$file.new" "$file" &&
        mv -f "$file.new" "$file"
    fi
done

echo "---"
echo "[Check files format]"
echo ""
file *

