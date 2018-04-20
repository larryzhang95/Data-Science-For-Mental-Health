# Data-Science-For-Mental-Health
Open Source Repository for code and programs I am developing for research in the Mental Health/Computational Neuroscience Domain.

Pipeline and Feature Extractions used for Mental Health/Computational Neuroscience Research

Please Run the following curl command to get checkpoint file.
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt

Key Contributions:
- model.py: contains all function(s) for acquiring/extracting data, flattening data, and applying data to model(s)
- feature_embeddings.py: contains all function(s) for extracting audio feature(s)
- example_demo.py: Example code to run on top of
- bipolar: example data from youtube audio to test out example_demo.py
- data.p: pickled preloaded data.

Credits to:
Google AudioSet: VGGish Model for Audio Feature Embeddings (see Vggish<insert name>.Py Files)<br />
NeuroLex Diagnostics: Original Demo Day Project, continuing research following Demo Day<br />
Dr. Reza Ghomi with UW DigiPsych Laboratory: For allowing me to pursue research within the domain of Computational Neuroscience<br />

Current Work in Progress:

- Extracting MRMR Feature Embeddings
- Extracting geMaps Feature Embeddings


Future Work:
- Add in NLP Model(s) in parallel to to Voice/Audio Model(s)
- Generalize Data Acquisition to include MRMR and geMaps features once completed.
