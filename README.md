# Recommender Algorithms Repository

This repository is dedicated to the re-implementation of various recommender algorithms from prominent research papers. The goal is to provide a modular and extensible framework for experimenting with these algorithms and comparing their performance across different datasets and use cases.

## Table of Contents
- [Overview](#overview)
- [Algorithms](#algorithms)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

Recommender systems play a crucial role in various domains, such as e-commerce, entertainment, and social media. This repository serves as a centralized collection of re-implemented recommendation algorithms, enabling researchers and practitioners to:
- Understand the implementation of various algorithms.
- Compare the performance of different models.
- Use these models as baselines or components for more advanced systems.

Each algorithm is implemented as a standalone module, ensuring clarity and ease of reuse.

## Algorithms

The repository currently includes implementations of the following algorithms:

- **A3NCF** (2018): An Adaptive Aspect Attention Model for Rating Prediction [Paper](https://www.ijcai.org/proceedings/2018/0521.pdf).
- **ALFM** (2018): Aspect-Aware Latent Factor Model [Paper](https://dl.acm.org/doi/pdf/10.1145/3178876.3186145).
- **ANR** (2018): Aspect-based Neural Recommender [Paper](https://raihanjoty.github.io/papers/chin-et-al-cikm-18.pdf).
- **AMF** (2018): Explainable recommendation with fusion of aspect information [Paper](https://yneversky.github.io/Papers/Hou2019_Article_ExplainableRecommendationWithF.pdf).
- **Att2Seq** (2017): Learning to Generate Product Reviews from Attributes [Paper](https://aclanthology.org/E17-1059.pdf) [Github](https://github.com/lileipisces/Att2Seq).
- **GMF** (2017): Generalized Matrix Factorization [Paper](https://arxiv.org/pdf/1708.05031).
- **MF** (2009): Matrix Factorization [Paper](https://www.cs.columbia.edu/~blei/fogm/2023F/readings/KorenBellVolinsky2009.pdf).
- **MLP** (2017): Multi-Layer Perceptron (Neural Collaborative Filtering) [Paper](https://arxiv.org/pdf/1708.05031).
- **NCF/NeuMF** (2017): Neural Collaborative Filtering [Paper](https://arxiv.org/pdf/1708.05031).
- **PEPLER** (2023): PersonalizedPrompt Learning for Explainable Recommendation [Paper](https://dl.acm.org/doi/pdf/10.1145/3580488).
- **PETER** (2021): Personalized Transformer for Explainable Recommendation [Paper](https://arxiv.org/pdf/2105.11601).
- **PMF** (2007): Probabilistic Matrix Factorization [Paper](https://proceedings.neurips.cc/paper_files/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf).

## Usage

### Running an Algorithm

1. Navigate to the directory of the desired algorithm (e.g., `PEPLER`).
2. Prepare the required data files, as described in the specific algorithm's `README.md` or comments.
3. Run the main script to execute the algorithm. For example:
   ```bash
   bash pepler.sh
   ```
   or
   ```python
   python main.py
   ```

### Adding a New Algorithm

To add a new algorithm:
1. Create a new folder named after the algorithm.
2. Implement the algorithm in a `main.py` file.
3. Add utility functions or modules in separate files, if necessary.
4. Provide a shell script (`.sh`) to simplify the execution process.

## Dependencies

To ensure smooth execution, the following dependencies are required:
- Python 3.8+
- NumPy
- PyTorch
- Pandas
- Cornac [Github](https://github.com/PreferredAI/cornac)
- Matplotlib (for visualizations)
- Nltk (for some models)

You can install all dependencies using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! If you have implemented a new recommendation algorithm or improved an existing one, feel free to:
1. Fork the repository.
2. Add your implementation following the structure and conventions.
3. Submit a pull request with a description of your changes.

## License

This repository is licensed under the MIT License. Feel free to use, modify, and distribute the code as per the license terms.
