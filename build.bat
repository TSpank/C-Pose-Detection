@echo off
REM ---- Clean previous build ----
@REM del /Q *.obj
@REM del /Q main.exe

REM ---- Compile ----
cl /EHsc /std:c++17 /O2 /Ob2 /DNDEBUG main.cpp kinematic_model.cpp mqttStickman.cpp MojoQuaternion.cpp MojoVector.cpp OneEuroFilter.cpp /I"." /I"C:\Users\ginga\vcpkg\installed\x64-windows\include" /link /LIBPATH:"C:\Users\ginga\vcpkg\installed\x64-windows\lib" paho-mqttpp3.lib paho-mqtt3c.lib ws2_32.lib /Fe:main.exe

pause
