
# HAT Testing Framework
[Back to Index ../](../index.md)


## Local Testing

For the OpenCL backend:

```bash
java @.ffi-opencl-test-suite
```

For the CUDA backend:

```bash
java @.ffi-cuda-test-suite
```

## Individual tests

```bash
java  @.ffi-cuda-test  CLASS#method
```

Passing a particular method to test is optional.

For example, to test the whole `TestArrays` class:

```bash
java  @.ffi-cuda-test hat.test.TestArrays

HAT Engine Testing Framework
Testing Backend: hat.backend.ffi.OpenCLBackend

Class: hat.test.TestArrays
Testing: #testHelloHat                  ..................... [ok]
Testing: #testVectorAddition            ..................... [ok]
Testing: #testVectorSaxpy               ..................... [ok]
Testing: #testSmallGrid                 ..................... [ok]

passed: 4, failed: 0, unsupported: 0
```

To test a single method (e.g., `testVectorAddition`):


```bash
java  @.ffi-cuda-test hat.test.TestArrays#testVectorAddition

HAT Engine Testing Framework
Testing Backend: hat.backend.ffi.OpenCLBackend

Class: hat.test.TestArrays
Testing: #testVectorAddition            ..................... [ok]

passed: 1, failed: 0, unsupported: 0
```

## Remote Testing

HAT provides its own testing framework that can be used to test on remote GPU servers.
First, you need to generate and configure the template:

```bash
bash scripts/remoteTesting.sh --generate-config-file
```

This flag generates a file in the local directory called `remoteTesting.conf`.
We just need to fill the template with the server names, and user names, fork to use, backends to test and the branch to use.

For instance:

```bash
# HAT Remote Testing Settings
SERVERS=server1 server2
REMOTE_USERS=juan juan
FORK=git@github.com:my-fork/babylon.git

#List of Backends to test
BACKENDS=ffi-cuda ffi-opencl

## Remote path. It assumes all servers use the same path
REMOTE_PATH=repos/babylon
## Branch to test
BRANCH=code-reflection
```

This assumes you have the `ssh-keygen` already configured.

Then, we need to build the project Babylon:

```bash
bash scripts/remoteTesting.sh --build-babylon
```

This builds babylon for each of the servers specified. The project is stored in the path specified in `REMOTE_PATH`.

Once it is finished, you can run the unit-tests on the remote GPU servers as follows:

```bash
bash scripts/remoteTesting.sh
```
