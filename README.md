# arrayfire-example
An example utilization of the arrayfire library. Used in the context of the seminar i'm writing on gpu programming.

You need to have arrayfire installed. In my case I did it from source, although there are
pre-compiled binaries on ubuntu. More info [here](http://arrayfire.org/docs/installing.htm).

## Python bindings

```
$ sudo apt install virtualenvwrapper
$ mkvirtualenv -p /usr/bin/python3 arrayfire
(arrayfire) $ pip install arrayfire
```
Afterwards, you need the following things in your path

- `/usr/local/arrayfire/lib`
- `/usr/local/cuda/lib64`
- `/usr/local/cuda/nvvm/lib64`
