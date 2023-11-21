# Kaggle Competition

## Workflow for creating new Submissions

1. Copy the `baseline.ipynb` Notebook.
2. Modify the "Embeddings" and "Create Model" Steps.
3. Go to [Kaggle Code](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/code) and click "New Notebook".
4. Click on "File" -> "Import Notbeook" and select your Notebook.
5. Disable Internet Access in "Notebook options".
6. Set the `is_submission` flag to True.
7. Make sure the Notebbok runs without Error and produces the same Output as the `sample_submission.csv` found in "Data".
8. Click "Submit" in "Submit To Competition".