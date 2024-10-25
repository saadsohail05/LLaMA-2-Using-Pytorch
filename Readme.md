
# 🦙 LLaMA 2 Implementation with PyTorch

This repository contains an **Educational Implementation of LLaMA 2** using PyTorch. It is inspired by the official code by Meta, recreated from scratch to deepen the understanding of transformer-based language models like LLaMA.

---

## 🧠 Overview

LLaMA 2 (Large Language Model Meta AI) is a cutting-edge transformer-based model designed for various natural language processing (NLP) tasks, such as text generation, text classification, and question answering. This repository showcases an end-to-end implementation of the LLaMA model using PyTorch, focusing on providing an educational resource for understanding the intricacies of model architectures.

## ✨ Features

- PyTorch implementation of the LLaMA 2 model.
- Transformer-based architecture with multi-head self-attention.
- Support for text generation with customizable parameters.
- SentencePiece tokenizer integration.
- Batched inference for efficient text completion.

---

## 🗂️ Project Structure

```
├── checkpoints/                # Folder for storing model checkpoints
├── Inference.py                # Main script for model inference
├── model.py                    # LLaMA model architecture
├── tokenizer.model             # SentencePiece tokenizer file
├── requirements.txt            # List of dependencies
└── README.md                   # This README file
```

### Key Files:

- **Inference.py**: Script to load the model and perform text completion.
- **model.py**: Contains the transformer model and utility functions.
- **checkpoints/**: Directory for model checkpoints (must include the `.pth` files and `params.json`).
- **tokenizer.model**: Tokenizer file for SentencePiece.

---

## 🚀 Installation & Setup

Follow the steps below to install and run the project.

### 1. Clone the repository
```bash
git clone https://github.com/saadsohail05/llama-2-pytorch.git
cd llama-2-pytorch
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running Inference

Once the environment is set up, use the following command to run inference and generate text with the LLaMA 2 model:

```bash
python Inference.py --checkpoint_path checkpoints/llama2.pth --prompt "Once upon a time"
```

This will produce text completions based on the given prompt.

### Example Output:
```
Once upon a time, in a distant land, there was a great kingdom that prospered...
```

---

## 📊 Training (Coming Soon)

Training scripts are under development and will be included in a future update.

---

## 🔧 Troubleshooting

- Ensure that your environment is properly activated before running any commands.
- Verify that the `checkpoints/` folder contains the necessary model files.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check out the [issues page](https://github.com/saadsohail05/llama-2-pytorch/issues).

---

## 📧 Contact

If you have any questions or feedback, feel free to reach out to me via [saadsohail5104@gmail.com](mailto:saadsohail5104@gmail.com).

---

