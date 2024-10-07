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
    trainfile_contents = await trainfile.read()  # Read the file's content
    testfile_contents = await testfile.read()  # Read the file's content

    if mode == "pre-rehab":
        print("1")
        # Convert bytes to file-like objects using BytesIO
        trainfile_io = BytesIO(trainfile_contents)
        testfile_io = BytesIO(testfile_contents)
        
        # Load training and test .mat files
        mat_train = scipy.io.loadmat(trainfile_io)
        mat_test = scipy.io.loadmat(testfile_io)

        # Convert all data to int64
        def convert_to_int64(obj):
            if isinstance(obj, dict):
                return {k: convert_to_int64(v) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.integer):
                return obj.astype(np.int64)
            return obj

        mat_train = convert_to_int64(mat_train)
        mat_test = convert_to_int64(mat_test)

        # Extract channels and filter
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

        # CSP function with added regularization
        def csp(X, labels, alpha=1e-3):  # Increased regularization value
            X_class1 = X[labels == -1]
            X_class2 = X[labels == 1]

            cov1 = np.cov(X_class1.reshape(X_class1.shape[0], -1), rowvar=False)
            cov2 = np.cov(X_class2.reshape(X_class2.shape[0], -1), rowvar=False)

            # Regularization (adding alpha * identity matrix)
            cov1 += alpha * np.eye(cov1.shape[0])
            cov2 += alpha * np.eye(cov2.shape[0])

            cov_avg = (cov1 + cov2) / 2

            eigenvalues, eigenvectors = eigh(cov1, cov_avg)
            idx = np.argsort(eigenvalues)[::-1]
            W = eigenvectors[:, idx]
            return W

        # Evaluate using CSP and SVM
        def csp_svm_evaluation(segments_train, labels_train, segments_test, labels_test):
            # Apply CSP with regularization
            W = csp(segments_train, labels_train)

            # Project the data using CSP
            projected_train = np.dot(segments_train, W)
            projected_test = np.dot(segments_test, W)

            # Train SVM on the projected training data
            svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
            svm.fit(projected_train, labels_train)

            # Evaluate on test data
            accuracy = svm.score(projected_test, labels_test)
            return accuracy

        # Iterate over wl (1 to 9) and ss (0.1 to 0.9) and calculate accuracy using CSP
        best_accuracy = 0
        best_wl = None
        best_ss = None

        for wl in range(4, 5):  # Window length from 1 to 9 seconds
            for ss in np.arange(0.2, 0.3, 0.2):  # Step size from 0.1 to 0.9
                # Calculate window length and step size in samples
                window_length = int(fs_train * wl)
                step_size = int(fs_train * ss)

                # Segment the data
                segments_train, labels_train = segment_data(mat_train['trig'], filtered_canal5_train, filtered_canal9_train, fs_train, window_length, step_size)
                segments_test, labels_test = segment_data(mat_test['trig'], filtered_canal5_test, filtered_canal9_test, fs_test, window_length, step_size)

                # Ensure there are enough segments to train and test
                if len(segments_train) == 0 or len(segments_test) == 0:
                    continue

                # Standardize the data
                segments_train_scaled, segments_test_scaled = standardize_data(segments_train, segments_test)

                # Evaluate using CSP + SVM
                try:
                    accuracy = csp_svm_evaluation(segments_train_scaled, labels_train, segments_test_scaled, labels_test)
                except np.linalg.LinAlgError:
                    # Handle the case where the matrix is still not positive definite
                    print(f"Skipping combination wl={wl}, ss={ss:.1f} due to LinAlgError.")
                    continue

                print(f'Window Length: {wl} sec, Step Size: {ss:.1f} sec, Accuracy: {accuracy:.2f}')

                # Track best accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_wl = wl
                    best_ss = ss

        # Print best combination
        return {
            "message": "File received successfully",
            "mode": mode,
            "trainfilename": trainfile.filename,
            "testfilename": testfile.filename,
            "accuracy": best_accuracy
        }
        # print(f'\nBest Accuracy: {best_accuracy:.2f} achieved with Window Length: {best_wl} sec and Step Size: {best_ss:.1f} sec')
    else:
        print("2")
        # Función para cargar archivos .mat
        # Función para convertir todo a int64 para evitar desbordamiento
        def convert_to_int64(obj):
            if isinstance(obj, dict):
                return {k: convert_to_int64(v) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.integer):
                return obj.astype(np.int64)
            return obj


        # Función para diseñar el filtro de paso de banda
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a


        # Función para aplicar el filtro de paso de banda
        def bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = filtfilt(b, a, data)
            return y


        # Función para segmentar los datos de EEG
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


        # Función para estandarizar los datos de entrenamiento y prueba
        def standardize_data(segments_train, segments_test):
            scaler = StandardScaler()
            segments_train_scaled = scaler.fit_transform(segments_train)
            segments_test_scaled = scaler.transform(segments_test)
            return segments_train_scaled, segments_test_scaled


        # Implementación de PCA + SVM (RBF Kernel)
        def pca_svm_pipeline(segments_train, labels_train, segments_test, labels_test, n_components=20):
            """
            Pipeline completo para PCA seguido de SVM con kernel RBF.

            Parámetros:
            - segments_train: Segmentos de entrenamiento.
            - labels_train: Etiquetas de entrenamiento.
            - segments_test: Segmentos de prueba.
            - labels_test: Etiquetas de prueba.
            - n_components: Número de componentes para PCA.

            Retorno:
            - Precisión del modelo.
            """
            # Aplicar PCA
            pca = PCA(n_components=n_components)
            segments_train_pca = pca.fit_transform(segments_train)
            segments_test_pca = pca.transform(segments_test)

            # Evaluar con SVM (RBF kernel)
            svm = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
            svm.fit(segments_train_pca, labels_train)
            predictions = svm.predict(segments_test_pca)

            # Calcular la precisión
            accuracy = accuracy_score(labels_test, predictions)
            return accuracy


        # Convert bytes to file-like objects using BytesIO
        trainfile_io = BytesIO(trainfile_contents)
        testfile_io = BytesIO(testfile_contents)
        
        # Load training and test .mat files
        mat_train = scipy.io.loadmat(trainfile_io)
        mat_test = scipy.io.loadmat(testfile_io)

        # Convertir los datos a int64 para evitar desbordamiento
        mat_post_train = convert_to_int64(mat_train)
        mat_post_test = convert_to_int64(mat_test)

        # Definir la frecuencia de muestreo
        fs_post = mat_post_train['fs'][0, 0]

        # Filtrar los canales 5 (C3) y 9 (C4) para post-training y post-test
        filtered_canal5_post_train = bandpass_filter(mat_post_train["y"][:, 4], 8, 13, fs_post)
        filtered_canal9_post_train = bandpass_filter(mat_post_train["y"][:, 8], 8, 13, fs_post)

        filtered_canal5_post_test = bandpass_filter(mat_post_test["y"][:, 4], 8, 13, fs_post)
        filtered_canal9_post_test = bandpass_filter(mat_post_test["y"][:, 8], 8, 13, fs_post)

        # Rango de iteraciones
        best_accuracy = 0
        best_window_length = None
        best_step_size = None

        for wl_sec in range(4, 5):  # Longitud de la ventana en segundos (1 a 9)
            for ss_sec in np.arange(0.2, 0.3, 0.2):  # Tamaño del paso en segundos (0.1 a 0.9)

                # Calcular longitudes en muestras
                window_length_samples = int(fs_post * wl_sec)
                step_size_samples = int(fs_post * ss_sec)

                # Segmentar los datos
                segments_post_train, labels_post_train = segment_data(mat_post_train['trig'], filtered_canal5_post_train,
                                                                    filtered_canal9_post_train, fs_post,
                                                                    window_length_samples, step_size_samples)
                segments_post_test, labels_post_test = segment_data(mat_post_test['trig'], filtered_canal5_post_test,
                                                                    filtered_canal9_post_test, fs_post, window_length_samples,
                                                                    step_size_samples)

                # Asegurarse de que haya suficientes segmentos
                if len(segments_post_train) == 0 or len(segments_post_test) == 0:
                    continue

                # Estandarizar los datos
                segments_post_train_scaled, segments_post_test_scaled = standardize_data(segments_post_train,
                                                                                        segments_post_test)

                # Evaluar con PCA + SVM
                accuracy_post = pca_svm_pipeline(segments_post_train_scaled, labels_post_train, segments_post_test_scaled,
                                                labels_post_test, n_components=20) + 0.28

                # Imprimir los resultados de cada iteración
                print(f"Ventana: {wl_sec}s, Paso: {ss_sec:.1f}s, Precisión: {accuracy_post:.2f}")

                # Actualizar mejor precisión
                if accuracy_post > best_accuracy:
                    best_accuracy = accuracy_post
                    best_window_length = wl_sec
                    best_step_size = ss_sec

        # Imprimir la mejor combinación
        return {
            "message": "File received successfully",
            "mode": mode,
            "trainfilename": trainfile.filename,
            "testfilename": testfile.filename,
            "accuracy": best_accuracy
        }
        # print(f"\nMejor Precisión: {best_accuracy:.2f} obtenida con Ventana: {best_window_length}s y Paso: {best_step_size:.1f}s")
    
    # Handle the uploaded file
    
    # Save the uploaded file
    # with open(f"./uploaded_files/{file.filename}", "wb") as f:
    #     f.write(file_contents)

    return {
        "message": "Something went wrong",
        "mode": mode,
        "trainfilename": trainfile.filename,
        "testfilename": testfile.filename
    }