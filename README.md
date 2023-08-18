# Stability Non-negative Matrix Factorization (staNMF)

This repository contains implementations for computing the stability of non-negative matrix factorizations (staNMF).

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Description of the Code](#description-of-the-code)
4. [Contributing](#contributing)
5. [License](#license)

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
```

2. Navigate to the cloned directory:
```bash
cd <repository_directory>
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

You can execute the provided code as a standard Python script:

```bash
python <script_name>.py
```

## Description of the Code

### staNMF

This class provides methods for computing non-negative matrix factorization (NMF) and evaluating its stability. Key features include:

- Loading data from a file or directly as a numpy array.
- Generating initial guesses for factorization.
- Performing NMF.
- Calculating instability metrics.
- Plotting the instability results.

### MNIST Data Processing Script

The second script uses the `staNMF` class to apply stability non-negative matrix factorization to the MNIST dataset. The steps include:

1. Loading the MNIST dataset.
2. Preprocessing the data using MinMax scaling.
3. Applying staNMF.
4. Plotting instability results for different values of K.

Parallel processing is utilized for different values of K, and results are aggregated for visualization.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
