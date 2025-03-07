Converting ONNX models to CoreML with ease.

A work in progress. 

`cd` into project directory, then `convert.py` to start.  

It will prompt if something can't be handled automatically.

In a [zipfile](https://github.com/Oil3/onnx-coreml-2025/blob/current/coremltoolsx/3signedLibraries.zip) I signed and notarized the libraries `libcoremlpython.so` `libmilstoragepython.so` `libmodelpackage.so`, you need to unpack them and put them back into the `coremltoolsx` folder from the `onnx-coreml-2025` folder.   
If there is an issue, you can copy your own libraries that you can find inside the oot of your existing `coremltools` python site-packages directory.     



I need to know what doesn't work so we can fix it, layer per layer.    
I mostly use Ultralytics 'Yolo' models, and there are many other ops I might even not know about. 
With coremltools 8.1, we can virtually convert everything.
The three main ideas here are: 
>make a model run -even if that means "resize bicubic" becomes "resize bilinear".
>it has to be easy
>every layer can be manually modified, changed or skipped


A mac is a mac and it needs to stay comfortable.
