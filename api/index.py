from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import scipy.io
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import traceback  # To print the full stack trace in case of errors

# Create FastAPI instance
app = FastAPI(docs_url="/api/py/docs", openapi_url="/api/py/openapi.json")

# Create the directory for uploaded files if it doesn't exist
os.makedirs("./uploaded_files", exist_ok=True)

# Allow CORS for your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/py/evaluate")
async def evaluate(trainfile: UploadFile = File(...),  testfile: UploadFile = File(...), mode: str = Form(...)):
    try:
        trainfile_contents = await trainfile.read()  # Read the file's content
        testfile_contents = await testfile.read()  # Read the file's content

        try:
            # Convert bytes to file-like objects using BytesIO
            trainfile_io = BytesIO(trainfile_contents)
            testfile_io = BytesIO(testfile_contents)

            # Load training and test .mat files
            mat_train = scipy.io.loadmat(trainfile_io)
            mat_test = scipy.io.loadmat(testfile_io)
        except Exception as e:
            print(f"Error loading .mat files: {e}")
            print(traceback.format_exc())  # Print full error stack trace
            return {"error": "Error loading .mat files"}

        try:
            # Convert all data to int64
            def convert_to_int64(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_int64(v) for k, v in obj.items()}
                elif isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.integer):
                    return obj.astype(np.int64)
                return obj

            mat_train = convert_to_int64(mat_train)
            mat_test = convert_to_int64(mat_test)
        except Exception as e:
            print(f"Error converting data to int64: {e}")
            print(traceback.format_exc())
            return {"error": "Error converting data to int64"}

        try:
            # Filter the data
            def butter_bandpass(lowcut, highcut, fs, order=5):
                nyq = 0.5 * fs
                low = lowcut / nyq
                high = highcut / nyq
                b, a = butter(order, [low, high], btype='band')
                return b, a

            def bandpass_filter(data, lowcut, highcut, fs, order=5):
                b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                y = filtfilt(b, a, data)
                return y

            # Define sampling frequency
            fs_train = mat_train['fs'][0, 0]
            fs_test = mat_test['fs'][0, 0]

            # Filter the training and test signals
            filtered_canal5_train = bandpass_filter(mat_train["y"][:, 4], 8, 13, fs_train)
            filtered_canal9_train = bandpass_filter(mat_train["y"][:, 8], 8, 13, fs_train)

            filtered_canal5_test = bandpass_filter(mat_test["y"][:, 4], 8, 13, fs_test)
            filtered_canal9_test = bandpass_filter(mat_test["y"][:, 8], 8, 13, fs_test)
        except Exception as e:
            print(f"Error filtering the data: {e}")
            print(traceback.format_exc())
            return {"error": "Error filtering the data"}

        try:
            # Segmentation function
            def segment_data(trig, canal5, canal9, fs, window_length, step_size):
                segments = []
                labels = []
                trig = trig.flatten().astype(np.int64)

                for start in range(0, len(trig) - window_length, step_size):
                    if trig[start] == -1:
                        segment = np.array([canal5[start:start + window_length]])
                        segments.append(segment)
                        labels.append(-1)
                    elif trig[start] == 1:
                        segment = np.array([canal9[start:start + window_length]])
                        segments.append(segment)
                        labels.append(1)

                segments = np.array(segments).reshape(len(segments), -1)
                labels = np.array(labels)
                return segments, labels

            # Standardization
            def standardize_data(segments_train, segments_test):
                scaler = StandardScaler()
                segments_train_scaled = scaler.fit_transform(segments_train)
                segments_test_scaled = scaler.transform(segments_test)
                return segments_train_scaled, segments_test_scaled
        except Exception as e:
            print(f"Error during segmentation or standardization: {e}")
            print(traceback.format_exc())
            return {"error": "Error during segmentation or standardization"}

        try:
            # Train and evaluate models (LDA + SVM)
            def lda_svm_evaluation(segments_train, labels_train, segments_test, labels_test):
                # LDA model
                lda = LinearDiscriminantAnalysis()
                lda.fit(segments_train, labels_train)

                # Predict probabilities for test set using LDA
                test_probabilities_lda = lda.predict_proba(segments_test)
                right_hand_prob_lda = test_probabilities_lda[:, 0] * 100  # Probability of right hand
                left_hand_prob_lda = test_probabilities_lda[:, 1] * 100   # Probability of left hand

                # SVM model
                svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
                svm.fit(segments_train, labels_train)

                # Predict probabilities for test set using SVM
                svm_probabilities = svm.predict_proba(segments_test)
                right_hand_prob_svm = svm_probabilities[:, 0] * 100  # Probability of right hand
                left_hand_prob_svm = svm_probabilities[:, 1] * 100   # Probability of left hand

                # Return both models' probabilities for further usage
                return right_hand_prob_lda, left_hand_prob_lda, right_hand_prob_svm, left_hand_prob_svm
        except Exception as e:
            print(f"Error training and evaluating LDA and SVM models: {e}")
            print(traceback.format_exc())
            return {"error": "Error training and evaluating models"}

        try:
            # Segmentation parameters
            window_length = int(fs_train * 4)  # 4 seconds window
            step_size = int(fs_train * 0.2)    # 0.2 seconds step size

            # Segment the data
            segments_train, labels_train = segment_data(mat_train['trig'], filtered_canal5_train, filtered_canal9_train, fs_train, window_length, step_size)
            segments_test, labels_test = segment_data(mat_test['trig'], filtered_canal5_test, filtered_canal9_test, fs_test, window_length, step_size)

            # Ensure there are enough segments
            if len(segments_train) == 0 or len(segments_test) == 0:
                return {"message": "Not enough test data to evaluate."}

            # Standardize the data
            segments_train_scaled, segments_test_scaled = standardize_data(segments_train, segments_test)

            # Evaluate LDA and SVM probabilities
            right_hand_prob_lda, left_hand_prob_lda, right_hand_prob_svm, left_hand_prob_svm = lda_svm_evaluation(segments_train_scaled, labels_train, segments_test_scaled, labels_test)
        except Exception as e:
            print(f"Error in the evaluation process: {e}")
            print(traceback.format_exc())
            return {"error": "Error in the evaluation process"}

        try:
            # Segments for x-axis
            segments = np.arange(1, len(labels_test) + 1)

            # Output summarized LDA and SVM probabilities for each segment
            for i in range(len(labels_test)):
                print(f"Segment {i+1}: LDA -> Right Hand: {right_hand_prob_lda[i]:.2f}%, Left Hand: {left_hand_prob_lda[i]:.2f}% | "
                      f"SVM -> Right Hand: {right_hand_prob_svm[i]:.2f}%, Left Hand: {left_hand_prob_svm[i]:.2f}%")

            return {
                "message": "File received successfully",
                "mode": mode,
                "trainfilename": trainfile.filename,
                "testfilename": testfile.filename,
                "lda_right_hand_probabilities": right_hand_prob_lda.tolist(),
                "lda_left_hand_probabilities": left_hand_prob_lda.tolist(),
                "svm_right_hand_probabilities": right_hand_prob_svm.tolist(),
                "svm_left_hand_probabilities": left_hand_prob_svm.tolist(),
            }
        except Exception as e:
            print(f"Error processing and returning results: {e}")
            print(traceback.format_exc())
            return {"error": "Error processing and returning results"}

    except Exception as e:
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())
        return {"error": "An unexpected error occurred"}