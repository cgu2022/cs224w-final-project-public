# Stanford CS224W Final Project: Adptable Course Recomendation System with the OnCourse Platform

William Huang, Chris Gu, and Xing Tan.

## Note

Because [OnCourse](https://oncourse.college) does not publicly release its data, we cannot include the dataset we used in the repo. This was discussed with TA's beforehand [here](https://edstem.org/us/courses/47423/discussion/3613443).

## Contents

Work from the Milestone:
- `milestone_work/basic_visualization.ipynb`: visualization of OnCourse enrollment data.
- `milestone_work/milestone.ipynb`: implementation of LightGCN and results on course recommendation task


Hand-curated Features:
- course description embeddings
    - `course_descrption_embeddings/embedding.ipynb`: generating OpenAI Ada embeddings for course descriptions and compressing them with an autoencoder
    - `course_description_embeddings/embedding_visualization.ipynb`: visualizing the OpenAI Ada embeddings
- year predictions
    - `predict_year/predict_year.ipynb`: generating probability distributions for which year a user is in, based on the courses they've pinned

Models:
- LightGCN implementation
    - `lightgcn/lightGCN_emb_32_12-10.ipynb`: implementation of LightGCN with all the features we describe in our Medium post
    - `lightgcn/eval_only.ipynb`: notebook to evaluate a trained model of LightGCN
- Heterogeneous GNN implementation
    - `heterogeneous/linkpred.py`: implementation of Heterogeneous GNN with features described in Medium post
    - `heterogeneous/linkpred.ipynb`: notebook explaining each portion of `linkpred.py`. Doesn't actually run, due to argparse portion.

