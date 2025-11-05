@echo off
REM ---- Clean previous build ----
del /Q *.obj
del /Q main.exe

REM ---- Compile ----
@REM cl /EHsc /std:c++17 main.cpp kinematic_model.cpp mqttStickman.cpp /I"." /I"C:\Users\ginga\vcpkg\installed\x64-windows\include" /link /LIBPATH:"C:\Users\ginga\vcpkg\installed\x64-windows\lib" paho-mqttpp3.lib paho-mqtt3c.lib ws2_32.lib /Fe:main.exe
cl /EHsc /std:c++17 main.cpp kinematic_model.cpp mqttStickman.cpp MojoQuaternion.cpp MojoVector.cpp /I"." /I"C:\Users\ginga\vcpkg\installed\x64-windows\include" /link /LIBPATH:"C:\Users\ginga\vcpkg\installed\x64-windows\lib" paho-mqttpp3.lib paho-mqtt3c.lib ws2_32.lib /Fe:main.exe

pause
