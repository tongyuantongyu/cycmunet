mkdir build
cd build

cmake -G Ninja -DVAPOURSYNTH_PLUGIN=ON -DCMAKE_BUILD_TYPE=Release ..
if %ERRORLEVEL% neq 0 exit 1
ninja vs-cycmunet
if %ERRORLEVEL% neq 0 exit 1

mkdir %PREFIX%\vapoursynth64\plugins
copy vs-cycmunet.dll %PREFIX%\vapoursynth64\plugins\vs-cycmunet.dll
