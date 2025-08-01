# Plant_Disease_Detection
Plant Disease Detection

📋 Project Overview
Our project focuses on creating an intuitive Streamlit application that empowers users to upload images of plant leaves for accurate identification of plant diseases. Utilizing a Convolutional Neural Network (CNN) model, this tool is designed to support farmers and gardeners in swiftly diagnosing plant health issues, thereby facilitating timely interventions to enhance crop management.

📂 Dataset
The dataset for our project comprises images categorized into 38 classes of plant species and their associated diseases, structured into training, validation, and testing subsets.

Key classes include:

Tomatoes (e.g., Late blight, Healthy, Early blight)
Grapes (e.g., Healthy, Black rot)
Oranges (e.g., Huanglongbing)
Potatoes (e.g., Healthy, Late blight)
Corn (maize) (e.g., Northern Leaf Blight)
Strawberries (e.g., Leaf scorch)
Other classes such as Peaches, Apples, Soybeans, Squash, Blueberries, and Cherries.
This structured dataset enables effective training and evaluation of machine learning models for plant disease detection.

📥 Download the Dataset
You can download the dataset from this link: https://docs.google.com/document/d/1cNLN_fNwurD_aluWaaMIZ-4hKar3FUq-A8Y-vYLEnaE/edit?tab=t.0

🎯 Objectives and Goals
Our project encompasses comprehensive development, including the establishment of an image upload interface using Streamlit, training the CNN model, and delivering a fully functional application that is easy to use. With a strong emphasis on real-world applicability, the project aims to equip farmers and gardeners with a rapid diagnosis tool, ultimately enabling them to respond effectively to plant diseases and optimize their agricultural practices.

🚀 Approach
1. Image Preprocessing
Implemented image preprocessing steps such as resizing, normalization, and augmentation to improve model performance. Utilized the New Plant Diseases Dataset from Kaggle, which contains images of plant leaves labeled with various diseases.

2. Disease Classification
Developed and trained a Convolutional Neural Network (CNN) model to classify plant diseases based on the uploaded images. Used the dataset from Kaggle for training and testing, applying techniques such as data augmentation and transfer learning to enhance model accuracy. Additionally, compared the performance of the Custom CNN model with at least three pretrained models to ensure the custom model outperforms existing ones. In our project, we have used these pre-trained models:

3. Performance Metrics
We are predicting the different plant diseases using our Custom CNN. Therefore, the following are the metrics that we used:

Accuracy: Measures the overall correctness of the predictions.

F1 Score: Balances performance across all classes by averaging F1 scores.

Precision: Measures how many of the positive predictions were correct.

Recall: Evaluates the ability of the model to capture all relevant positive cases.

🚧 Challenges and Future Work
Challenges:
In plant disease detection, one significant challenge is handling imbalanced datasets, where some diseases have fewer images for training. This imbalance can bias the model towards more common classes, resulting in poor performance on rare diseases. While the provided dataset was already augmented, implementing image preprocessing steps such as resizing, normalization, and further augmentation could enhance model performance. Moreover, optimizing the application for real-time inference is critical, as users expect quick and accurate predictions. Achieving low latency requires careful optimization of both the model architecture and the application infrastructure, ensuring seamless user experiences without compromising prediction accuracy.

Future Work:
Future enhancements could include implementing a feedback mechanism for users to report incorrect predictions, providing valuable data for model improvement and fostering community engagement. Furthermore, expanding the model to recognize a wider range of plant species and diseases is crucial to meet the diverse needs of users in agriculture. By continually updating the dataset and refining the model, the application can become a vital tool for farmers and gardeners in managing plant health effectively.

🌱 Results
The project aims to deliver a fully functional Streamlit-based web application that allows users to upload images of plant leaves and receive accurate predictions about plant diseases. A detailed model performance report will accompany the application, highlighting the accuracy and other evaluation metrics of the CNN model. Additionally, the project will include a user guide that explains how to operate the application and interpret the results effectively.
