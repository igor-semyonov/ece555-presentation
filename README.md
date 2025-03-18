Presentation for ECE 555

# Notes

I have managed to get the add example in the Rust_CUDA crate working.
I ran into some issues and the devs helped: https://github.com/Rust-GPU/Rust-CUDA/issues/163
I was running a newer version of CUDA than expected and they updated the way the crate checks for versions to fix it.

The main thing I had to do is compile LLVM and set and enviroment variable pointing to that LLVM.

1. Go to the [release page](https://releases.llvm.org/download.html#7.0.1) for LLVM 7.0.1. It has to be LLVM 7, as that is the version nvvm supports. I don't know if 7.1 works. I tested 7.0.1.
2. I used the first link to download the source code.
3. Extract it using `tar -xvf <path-to-tarball>`
4. Create a build directory, such as `llvm-7.0.1.build` and `cd` into it.
5. Run the following cmake command to prepare the build.

```
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/home/kdcadet/code/llvm-7.0.1 -DCMAKE_BUILD_TYPE=Release ../llvm-7.0.1.src
```

Edit the arguments accordingly.

6. I compiled using `make -j24 && make install`, but you can also use cmake by running `cmake --build . && cmake --build . --target install`.

[LLVM build instructions](https://llvm.org/docs/CMake.html)

7. Now change to the llvm install directory and run `readlink -f bin/llvm-config`

This will print the full path to the `llvm-config` binary. Copy this for the next step.

8. In the shell where you plan to compile a Rust_CUDA project, run `export LLVM_CONFIG=<path-to-llvm-config-binary>`
For me, this is 
`export LLVM_CONFIG=/home/kdcadet/code/llvm-7.0.1/bin/llvm-config`

# Ideas for coding portion

- Monte Carlo Pi estimation: Generate points in the square of size 2x2 centered at the origin. The proportion inside the unit circle will give us an estimate of pi.
- Image convolution. A bit boring, but also should be relatively easy to do. We could also expand this to a small-ish CNN, maybe with backprop too.

