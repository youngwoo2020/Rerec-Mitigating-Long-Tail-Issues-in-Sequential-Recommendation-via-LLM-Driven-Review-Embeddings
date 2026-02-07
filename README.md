# RERec: Mitigating Long-Tail Issues in Sequential Recommendation via LLM-Driven Review Embeddings

## 📝 Abstract & Key Contributions
Sequential Recommendation Systems (SRS) often struggle with **data sparsity**, particularly for **cold-start users** and **long-tail items**. **RERec** (Review Enhanced Recommendation) bridges the gap between semantic understanding from LLMs and collaborative learning.

* **Review-based Semantic Integration:** Leverages LLM-based embeddings derived from representative review sentences to enhance item representations.
* **MMR-driven Selection:** Employs the Maximal Marginal Relevance (MMR) algorithm to select the most semantically diverse and cohesive review sentences for each item.
* **Balanced Fusion:** Integrates qualitative semantic attributes with quantitative interaction signals, significantly improving performance in sparse recommendation environments.
<img width="631" height="313" alt="image" src="https://github.com/user-attachments/assets/cdb78e96-2c44-42cf-a2bd-3bd1073e8ef6" />




---

## 📂 Dataset Preparation
The experiments utilize the **Amazon Review Data (2023)**.
1.  **Download:** Access the [Amazon Reviews'23](https://amazon-reviews-2023.github.io/) and download the raw Review and Metadata files.
2.  **Path:** Place the downloaded raw files into the `/raw` directory within this project.

---

## 🏃 Execution Pipeline
To reproduce the experimental results, please execute the scripts in the following numerical order:

### Phase 1: Data Preparation & Preprocessing
1.  **`1.load_data.py`**: Loads raw Amazon data from the `raw/` folder and maps review metadata.
2.  **`2.split_data.py`**: Splits the review data into specific time-based sequences.
3.  **`3.data_preprocess.py`**: Performs core preprocessing, including k-core filtering to ensure data quality.

### Phase 2: Semantic Review Extraction
4.  **`4.MMR.py`**: Applies the **MMR algorithm** to select the most representative review sentences for each item.
5.  **`5.data_filtering.py`**: Filters the interaction sequences based on the selected representative review data.

### Phase 3: File Management & LLM Embedding
6.  **`7.convert.py`** & **`8.cleaning.py`**: Utility scripts for renaming files and organizing the processed dataset.
7.  **`8.get_item_embedding.ipynb`**: Generates item-level semantic embeddings. (**Requires OpenAI API Key**).
8.  **`9.get_user_embedding.ipynb`**: Generates user-level semantic embeddings. (**Requires OpenAI API Key**).

### Phase 4: Model Training & Evaluation
9.  **`10.Rerec.py`**: Executes the final pipeline including **PCA** for dimensionality reduction, **Retriever User** creation, and running the training bash files.

---

## 🛠️ Requirements
- Python 3.8+
- PyTorch 2.x
- **OpenAI API Key:** Must be configured in the `.ipynb` files to generate embeddings.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
