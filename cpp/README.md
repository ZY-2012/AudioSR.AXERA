# C++ runtime

The C++ AX650 runtime is currently a scaffold. The complete, tested runtime for
this release is the Python SDK in `python/` (`axengine` + NumPy/librosa DSP).

`cpp/` keeps the cross-compilation entry points and the planned runtime layout:

- `main.cpp` — audio I/O + model chain
- `audiosr.hpp` — model sessions / DDIM / vocoder split
- `CMakeLists.txt`, `build_ax650.sh` — aarch64 AX650 build

Until the C++ port is completed, use:

```bash
python python/demo.py samples/listen/16k/speech_2_input_16k.wav -o out.wav
```

The Python SDK is the reference implementation and matches the reported
0.9917 full-chain similarity on AX650.
