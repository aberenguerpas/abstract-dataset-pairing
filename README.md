# abstract-dataset-pairing


### Corpus EDA
N° tables 1187 <br />
N° rows 339324849 <br />
N° cols 231828
N° numeric cols 220425 <br />
N° categorical cols 11403 <br />

|       | n_words_abstract | n_rows   | n_cols   | numeric_cols | categorical_cols |
|-------|------------------|----------|----------|--------------|------------------|
| count | 1187             | 1187     | 1187     | 1187         | 1187             |
| mean  | 116.142          | 285876   | 195.306  | 185.699      | 9.607            |
| std   | 188.266          | 3281804  | 2170.201 | 2170.298     | 47.393           |
| min   | 1                | 0        | 0        | 0            | 0                |
| 25%   | 18               | 75       | 6        | 2            | 1                |
| 50%   | 49               | 436      | 12       | 6            | 2                |
| 75%   | 160              | 4024     | 31       | 16           | 7                |
| max   | 2001             | 76245109 | 50282    | 50281        | 1266             |

### Alpha variations results
/alpha = 0 -> Only content
/alpha = 1 -> Only headers
| /alpha| SentenceBert |    bert    |  roberta  |    Bloom   |   w2v    |SciBert |   fst  |
|-------|--------------|------------|-----------|------------|----------|--------|--------|
|   0   |   0.140664   |  0.657940  | 0.203158  |            | 0.219585 |0.737520|0.533047|
| 0.1   |   0.149022   |  0.661316  | 0.208560  | 185.699    | 0.237718 |0.734428|0.549144|
| 0.2   |   0.157379   |  0.664691  | 0.213963  | 2170.298   | 0.255851 |0.731336|0.565242|
| 0.3   |   0.165736   |  0.668067  | 0.219365  | 0          | 0.273984 |0.728244|0.581339|
| 0.4   |   0.174093   |  0.671443  | 0.224768  | 2          | 0.292118 |0.725152|0.597436|
| 0.5   |   0.182450   |  0.674819  | 0.230170  | 6          | 0.310251 |0.722060|0.613534|
| 0.6   |   0.190807   |  0.678195  | 0.235573  | 16         | 0.328384 |0.718968|0.629631|
| 0.7   |   0.199165   |  0.681571  | 0.240975  | 50281      | 0.346517 |0.715876|0.645729|
| 0.8   |   0.207522   |  0.684947  | 0.246378  | 50281      | 0.364650 |0.712784|0.661826|
| 0.9   |   0.215879   |  0.688323  | 0.251781  | 50281      | 0.382783 |0.709692|0.677923|
| 1.0   |   0.224236   |  0.691699  | 0.257183  | 50281      | 0.400917 |0.706600|0.694021|