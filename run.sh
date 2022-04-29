echo "build fftw"

./configure

make

make install


echo "Building..."
make

echo "Runing..."
./compare -n 16 > result.log