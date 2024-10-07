## EEG Signal Processing for Chronic Stroke Patients
This project aims to optimize EEG pre-processing, feature extraction, and classification to detect arm movements in chronic stroke patients. We implement advanced techniques like band-specific filtering and Common Spatial Patterns (CSP) to improve motor imagery detection and support neurorehabilitation, which can be adapted to different patient datasets. 

## By combining:
- Band-pass filtering: Isolating relevant frequency bands for motor imagery detection.
- Common Spatial Patterns (CSP): Extracting spatial features that distinguish different motor tasks.
- Classification: Optimizing machine learning models to improve performance.


We used advanced techniques to have the best results, we combined CSP and SVM to have more accurate results, later on we changed to SVM and LDA to have faster results and still didn't lose much accuracy.


We addressed the variability found in EEG signals among chronic stroke patients and emphasized tailoring the techniques to individual datasets for better results.

## Key Features:
EEG Preprocessing: Removing noise and extracting frequency-specific information.

Classification: Training classifiers like LDA and SVM to detect arm movement intentions.

The results indicate that our approach improves the detection of motor imagery and shows promise for enhancing neurorehabilitation strategies.

## Example Output
The output includes classification results, such as accuracy, precision, and recall, based on the EEG data provided.

## Results
The project demonstrates that tailored EEG processing pipelines can significantly improve motor imagery detection in stroke patients. By using advanced filtering, CSP, and classifiers, the accuracy of detecting arm movements based on EEG signals improved across multiple datasets.

## Tech Stack
- Model: scipy, numpy, scikit-learn, python-multipart.
- Data Science Algorithms: CSP + SVM, CSP + VTLDA.
- Frontend: fastapi, nextjs

## Authors:

- Adolfo Hernández.            
- Roberto Morales.
- José Emilio Inzunza.             
- Esteban Ochoa.
- Esteban Muñoz.
- Alonso Rivera.
- Isabella Hurtado.
