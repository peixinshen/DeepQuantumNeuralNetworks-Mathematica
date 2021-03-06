# DeepQuantumNeuralNetworks-Mathematica

[![Language](https://img.shields.io/static/v1?label=Wolfram&labelColor=gray&message=Mathematica&color=d21c00&logo=wolfram-language&logoColor=white)](https://www.wolfram.com/mathematica/)
[![View notebooks](https://wolfr.am/HAAhzkRq)](https://wolfr.am/T3TVfBEh)

This *Mathematica* notebook can be used to classically simulate deep quantum neural networks as proposed in 
> K. Beer, D. Bondarenko, T. Farrelly, T. J. Osborne, R. Salzmann, and R. Wolf. [Training deep quantum neural networks](https://doi.org/10.1038/s41467-020-14454-2). Nat Commun 11, 808 (2020). 

Compared to [the original repo](https://github.com/qigitphannover/DeepQuantumNeuralNetworks) of the authors, this code has been rewritten in a more *Mathematica*-like fashion. In addition, its efficiency has also been dramatically improved, as we introduce a new function for the [Partial Trace](https://en.wikipedia.org/wiki/Partial_trace):

```mathematica
PartialTrace[densityMatrix_, traceList_, quditDim_ : 2] := 
    Module[{t = Flatten@{traceList}, qubitNum = Log[quditDim, Length@densityMatrix]}, 
        ArrayReshape[
            TensorContract[
                ArrayReshape[
                    densityMatrix, Table[quditDim, 2 qubitNum]
                ], Table[{i, i + qubitNum}, {i, t}]
            ], {quditDim^(qubitNum - Length@t), quditDim^(qubitNum - Length@t)}
        ]
    ]
```

This is the very initial version, detailed comments of the code will be updated in the near future.