# Dynamic Learning Technique

Dynamic learning technique allows the user to train a model in batch wise manner


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Exchange Rate Api

```bash
pip install dlt
```

## Usage

The DLT takes 2 argument with 6 optional arguments

```python
# Initialize an object to DLT
from DLT import *
from sklearn.tree import DecisionTreeRegressor

DLT(['X dataset'], ['Y dataset'], DecisionTreeRegressor())
```

## Features

- ## **Exception**
  New exceptions has been included
    - NoArgumentException
    - InvalidMachineLearningModel
    - InvalidDatasetProvided

Test cases has been included

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://github.com/TONYSTARK-EDITH/Dynamic-Learning-Technique/blob/9cf56eb5b1421e70d895bcf52e4f3c4964987df2/LICENSE)